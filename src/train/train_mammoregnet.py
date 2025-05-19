import time
import wandb
from tqdm import tqdm

from src.models.MammoRegNet import (AffineTransformer_block, MammoRegNet,
                                    SpatialTransformer_block)
from src.train.losses_mammoregnet import NCC, Regu_loss
from src.utils.utils import *


def train_val(
    train_loader,
    valid_loader,
    device,
    learning_rate,
    weight_decay,
    num_epochs,
    path_loggger,
    path_model,
    path_out,
    id,
    use_scheduler,
    max_iterations,
):
    print("[INFO] Starting training...")
    start_time = time.time()

    # Initialize logger
    logger = create_logger(path_loggger)
    logger.info(f"Training for {num_epochs} epochs")

    # Initialize model
    model = MammoRegNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Optional learning rate scheduler
    scheduler = None
    if use_scheduler == "True":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.97)
        logger.info("Using ExponentialLR scheduler")

    # Initialize WandB
    wandb.init(
        project="csaw_registration",
        config={
            "optimizer": "Adam",
            "architecture": "MammoRegNet",
            "dataset": "CSAW_CC",
            "epochs": num_epochs,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
        },
    )
    wandb.define_metric("epoch", hidden=True)
    for name in [
        "Training Total Loss", "Training Loss Segmentation (MSE)",
        "Training Loss NCC", "Training Loss NCC only affine",
        "Training Loss Regularisation", "Validation Loss NCC ",
        "Validation Loss NCC only affine", "Validation NJD"
    ]:
        wandb.define_metric(name, step_metric="epoch")

    # Loss functions
    Loss_ncc = NCC
    Loss_regu = Regu_loss
    lambda_regu = 1.0
    best_val_ncc = 1.0

    # Spatial transformers
    SpatialTransformer = SpatialTransformer_block(mode="nearest")
    SpatialTransformer().to(device).eval()
    AffineTransformer = AffineTransformer_block(mode="nearest")
    AffineTransformer().to(device).eval()


    # Training/Validation Loops
    for epoch in tqdm(range(num_epochs), desc="Epoch"):
        torch.cuda.empty_cache()
        logger.info(f"Epoch {epoch}/{num_epochs}")

        model.train()
        total_loss, regu_loss_sum, ncc_loss_sum, ncc_affine_sum = 0.0, 0.0, 0.0, 0.0
        counter = 0

        for batch_idx, batch in enumerate(train_loader):
            if counter >= max_iterations:
                break
            counter += 1

            img_fix = batch["img_fix"].to(device)
            img_mov = batch["img_mov"].to(device)

            pred = model(img_fix, img_mov)

            ncc_final = Loss_ncc().loss(img_fix, pred[0])
            ncc_affine = Loss_ncc().loss(img_fix, pred[2])
            ncc_loss = (1 - ncc_affine) + (1 - ncc_final)

            regu_loss = Loss_regu(pred[1])
            loss = ncc_loss + lambda_regu * regu_loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            regu_loss_sum += regu_loss.item()
            ncc_loss_sum += ncc_loss.item()
            ncc_affine_sum += (1 - ncc_affine).item()

        if scheduler:
            scheduler.step()

            # Epoch averages
        total_loss /= counter
        regu_loss_sum /= counter
        ncc_loss_sum /= counter
        ncc_affine_sum /= counter

        # Logging
        wandb.log({
            "epoch": epoch,
            "Training Total Loss": total_loss,
            "Training Loss Regularisation": regu_loss_sum,
            "Training Loss NCC": ncc_loss_sum,
            "Training Loss NCC only affine": ncc_affine_sum
        })
        logger.info(f"Losses - Total: {total_loss:.4f}, NCC: {ncc_loss_sum:.4f}, Regu: {regu_loss_sum:.4f}")

        logger.info("Training Total Loss: {}".format(total_loss))
        logger.info("Training Loss NCC: {}".format(ncc_loss_sum))
        logger.info("Training Loss Regularisation: {}".format(regu_loss_sum))

        print("Epoch", epoch)
        print("Training Total Loss", total_loss)
        print("Training Loss NCC", ncc_loss_sum)
        print("Training Loss Regularisation", regu_loss_sum)

        model.eval()
        with torch.no_grad():
            val_total_ncc, val_ncc_affine, val_njd_total, val_counter = 0.0, 0.0, 0.0, 0.0
            for idx, batch_val in enumerate(valid_loader):
                if val_counter >= max_iterations:
                    break
                val_counter += 1

                img_fix_val = batch_val["img_fix"].to(device)
                img_mov_val = batch_val["img_mov"].to(device)

                pred = model(img_fix_val, img_mov_val)

                ncc_final = Loss_ncc().loss(img_fix_val, pred[0])
                ncc_affine = Loss_ncc().loss(img_fix_val, pred[2])
                ncc_loss = (1 - ncc_affine) + (1 - ncc_final)

                val_total_ncc += ncc_loss.item()
                val_ncc_affine += (1 - ncc_affine).item()

                flow = pred[1].detach().cpu().permute(0, 2, 3, 1).numpy()
                val_njd_total += NJD_percentage(flow)

            # Calculate average metrics for the epoch
            val_ncc_loss = val_total_ncc / val_counter
            val_ncc_affine_loss = val_ncc_affine / val_counter
            val_njd = val_njd_total / val_counter

            # Save best model based on NCC
            if val_ncc_loss < best_val_ncc:
                best_val_ncc = val_ncc_loss
                best_model_path = os.path.join(path_out, f"best_model_registration_id-{id}.pth")
                checkpoint(model, best_model_path)
                logger.info(f"Saved best model at epoch {epoch} to {best_model_path}")

            # Log metrics
            wandb.log({
                "epoch": epoch,
                "Validation Loss NCC": val_ncc_loss,
                "Validation Loss NCC only affine": val_ncc_affine_loss,
                "Validation NJD": val_njd,
            })

            logger.info(f"Validation Loss NCC: {val_ncc_loss:.4f}")
            logger.info(f"Validation Loss NCC only affine: {val_ncc_affine_loss:.4f}")
            logger.info(f"Validation NJD: {val_njd:.4f}")

            print(f"[Epoch {epoch}] Validation Loss NCC: {val_ncc_loss:.4f} | Affine: {val_ncc_affine_loss:.4f} | NJD: {val_njd:.4f}")

    ##################      Finalization      ###################
    total_time = time.time() - start_time
    logger.info(f"Total training time: {total_time:.2f}s")
    print(f"[INFO] Total training time: {total_time:.2f}s")

    # Save final model
    print("[INFO] Saving final model...")
    torch.save(model.state_dict(), path_model)

    artifact = wandb.Artifact("model", type="model")
    artifact.add_file(path_model)
    wandb.log_artifact(artifact)
    wandb.finish()
