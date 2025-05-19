import time
import torch.nn as nn
import wandb
from tqdm import tqdm

from src.models.MammoRegNet import MammoRegNet
from src.models.model_combined_alignment_risk import (
    CombinedAlignmentRiskModel, CombinedImgAlignmentRiskModel,
    CombinedImgAlignmentRiskModel_downsample_img_deformation_field,
    RiskModelNoAlignment)
from src.utils.c_index import get_censoring_dist
from src.utils.utils import *


def train_val_jointly(
    train_loader,
    valid_loader,
    device,
    learning_rate,
    weight_decay,
    num_epochs,
    path_loggger,
    path_model,
    id,
    use_scheduler,
    out_dir,
    accumulation_steps,
    patience_lr_scheduler,
    patience,
    use_reg_loss,
    lambda_regu,
    lr_decay,
    no_feat_Alignment,
):
    print("[INFO] Training the network...")
    start_time = time.time()

    # Initialize logger
    logger = create_logger(path_loggger)
    logger.info(f"Number of Training Epochs: {num_epochs}")

    # Initialize model
    model_cls = RiskModelNoAlignment if no_feat_Alignment == "True" else CombinedAlignmentRiskModel
    model_risk = model_cls(num_years=5).to(device)

    optimizer = torch.optim.Adam(model_risk.parameters(), lr=learning_rate, weight_decay=weight_decay)

    scheduler = None
    if use_scheduler == "True":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=lr_decay,
            patience=patience_lr_scheduler,
            verbose=True,
        )
        print("Scheduler Initialized:", scheduler)

    # Initialize WandB
    wandb.init(
        project="EMBED_Risk_Prediction",
        config={
            "optimizer": optimizer.__class__.__name__,
            "architecture": "TemporalRiskPrediction",
            "dataset": "EMBED",
            "epochs": num_epochs,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "model": model_risk.__class__.__name__,
        },
    )

    # Define metrics
    wandb.define_metric("epoch", hidden=True)
    for metric in [
        "Training Loss", "Training Accuracy", "Training Alignment Loss", "Training Risk Loss",
        "Validation Validation Risk Loss", "Validation Alignment L2 Loss",
        "Training C-index", "Validation C-index",
        "Year 1 AUC", "Year 2 AUC", "Year 3 AUC", "Year 4 AUC", "Year 5 AUC"
    ]:
        wandb.define_metric(metric, step_metric="epoch")

    # Prepare losses
    alignment_loss_fn = nn.MSELoss()  # L2 loss for alignment
    Loss_regu = Regu_loss

    # Early Stopping
    best_c_index = 0
    patience_counter = 0

    # Training/Validation Loops
    for epoch in tqdm(range(num_epochs), desc="Epochs"):
        print(f"[INFO] Epoch: {epoch}")
        logger.info(f"##### Epoch: {epoch} #####")

        model_risk.train()
        running = {
            "loss": 0.0,
            "alignment_loss": 0.0,
            "alignment_loss_before": 0.0,
            "risk_loss": 0.0,
        }

        counter = 0
        all_preds, all_times, all_events = [], [], []

        for idx, batch in enumerate(train_loader):
            torch.cuda.empty_cache()
            counter += 1

            # Move data to device
            current_image = batch["current_image"].to(device, dtype=torch.float32)
            prior_image = batch["previous_image"].to(device, dtype=torch.float32)
            time_gap = batch["time_gap"].to(device)
            event_times_batch = batch["event_times"].to(device, dtype=torch.float32)
            event_observed_batch = batch["event_observed"]
            target = batch["target"]
            target_prior = batch["target_prior"]
            y_mask = batch["y_mask"]
            y_mask_prior = batch["y_mask_prior"]

            # Forward pass
            outputs = model_risk(current_image, prior_image, time_gap)
            del current_image, prior_image

            # Alignment loss
            if no_feat_Alignment != "True":
                aligned_prior = outputs["aligned_prior_feature"]
                current_features = outputs["current_feature"]
                prior_feature = outputs["prior_feature_before_alignment"]

                alignment_loss = alignment_loss_fn(aligned_prior, current_features)
                alignment_loss_before = alignment_loss_fn(prior_feature, current_features)
                running["alignment_loss"] += alignment_loss.item()
                running["alignment_loss_before"] += alignment_loss_before.item()

            # Risk loss
            pred = outputs["risk_prediction"]
            risk_loss_fused = get_risk_loss_BCE(pred["pred_fused"], target, y_mask)
            risk_loss_cur = get_risk_loss_BCE(pred["pred_cur"], target, y_mask)
            risk_loss_pri = get_risk_loss_BCE(pred["pred_pri"], target_prior, y_mask_prior)

            risk_loss = risk_loss_fused + risk_loss_cur + risk_loss_pri
            running["risk_loss"] += risk_loss.item()

            # Total loss
            if no_feat_Alignment == "True":
                total_loss = risk_loss
            else:
                regu_loss = Loss_regu(outputs["deformation_field"].permute(0, 2, 3, 1)) if use_reg_loss == "True" else 0
                total_loss = risk_loss + (alignment_loss / 100) + (
                    lambda_regu * regu_loss if use_reg_loss == "True" else 0)

            # Gradient accumulation
            total_loss = total_loss / accumulation_steps
            total_loss.backward()

            if (idx + 1) % accumulation_steps == 0 or (idx + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()

            running["loss"] += total_loss.item()
            all_preds.append(pred["pred_fused"].detach().cpu().numpy())
            all_times.append(event_times_batch.cpu().numpy())
            all_events.append(event_observed_batch.cpu().numpy())

        # Evaluation metrics
        preds = np.concatenate(all_preds, axis=0)
        times = np.concatenate(all_times, axis=0)
        events = np.concatenate(all_events, axis=0)
        censoring = get_censoring_dist(times, events)
        c_index = concordance_index_ipcw(times, preds, events, censoring)

        avg_loss = running["loss"] / counter
        avg_align = running["alignment_loss"] / counter
        avg_risk = running["risk_loss"] / counter

        wandb.log({
            "epoch": epoch,
            "Training C-index": c_index,
            "Training Loss": avg_loss,
            "Training Alignment Loss": avg_align,
            "Training Risk Loss": avg_risk,
        })

        print(f"[Epoch {epoch}] Total Loss: {avg_loss:.4f} | Alignment Loss: {avg_align:.4f} | Risk Loss: {avg_risk:.4f}")

        ##################      Validation      ###################

        with torch.inference_mode():
            model_risk.eval()
            valid_loss_total = 0.0
            alignment_loss_total = 0.0
            alignment_loss_before_total = 0.0
            risk_loss_total = 0.0

            predictions, event_observed, event_times = [], [], []
            num_batches = len(valid_loader)

            for batch in valid_loader:
                torch.cuda.empty_cache()

                # Extract batch data
                curr_img = batch["current_image"].to(device).float()
                prior_img = batch["previous_image"].to(device).float()
                time_gap = batch["time_gap"].to(device)
                event_times_batch = batch["event_times"]
                event_observed_batch = batch["event_observed"]
                y_mask = batch["y_mask"]
                y_mask_prior = batch["y_mask_prior"]
                target = batch["target"]
                target_prior = batch["target_prior"]

                # Model forward pass
                outputs = model_risk(curr_img, prior_img, time_gap)
                del curr_img, prior_img

                # Alignment Loss
                if no_feat_Alignment != "True":
                    aligned_prior = outputs["aligned_prior_feature"]
                    current_feat = outputs["current_feature"]
                    prior_feat = outputs["prior_feature_before_alignment"]

                    alignment_loss = alignment_loss_fn(aligned_prior, current_feat)
                    alignment_loss_before = alignment_loss_fn(prior_feat, current_feat)

                    alignment_loss_total += alignment_loss.item()
                    alignment_loss_before_total += alignment_loss_before.item()

                # Risk Loss
                risk_preds = outputs["risk_prediction"]
                loss_fused = get_risk_loss_BCE(risk_preds["pred_fused"], target, y_mask)
                loss_cur = get_risk_loss_BCE(risk_preds["pred_cur"], target, y_mask)
                loss_pri = get_risk_loss_BCE(risk_preds["pred_pri"], target_prior, y_mask_prior)
                risk_loss = loss_fused + loss_cur + loss_pri
                risk_loss_total += risk_loss.item()

                # Total loss
                total_loss = risk_loss + (alignment_loss / 100) if no_feat_Alignment != "True" else risk_loss
                valid_loss_total += total_loss.item()

                # Store predictions and labels
                predictions.append(risk_preds["pred_fused"].cpu().numpy())
                event_observed.append(event_observed_batch.cpu().numpy())
                event_times.append(event_times_batch.cpu().numpy())

            # Aggregate metrics
            predictions = np.concatenate(predictions, axis=0)
            event_times = np.concatenate(event_times, axis=0)
            event_observed = np.concatenate(event_observed, axis=0)

            # Compute AUCs
            aucs = compute_auc_x_year_auc(predictions, event_times, event_observed)
            for year, auc in aucs.items():
                print(f"Year {year + 1}: AUC = {auc:.4f}")
                wandb.log({f"Year {year + 1} AUC": auc, "epoch": epoch})

            # Compute validation C-index
            censoring_dist = get_censoring_dist(times, events)
            c_index_val = concordance_index_ipcw(event_times, predictions, event_observed, censoring_dist)

            # Scheduler update
            if use_scheduler == "True":
                scheduler.step(c_index_val)

            # Early stopping
            if c_index_val > best_c_index:
                best_c_index = c_index_val
                patience_counter = 0
                best_model_path = os.path.join(out_dir, f"best_model_risk_prediction_id-{id}.pth")
                torch.save(model_risk.state_dict(), best_model_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    early_stop_path = os.path.join(out_dir, f"early_stopping_risk_prediction_id-{id}.pth")
                    torch.save(model_risk.state_dict(), early_stop_path)
                    print("Early stopping triggered.")
                    break

            # Logging to WandB and console
            alignment_loss_avg = alignment_loss_total / num_batches
            risk_loss_avg = risk_loss_total / num_batches

            wandb.log({
                "Validation Alignment L2 Loss": alignment_loss_avg,
                "Validation Risk Loss": risk_loss_avg,
                "Validation C-index": c_index_val,
                "epoch": epoch,
            })

            logger.info(f"Validation Alignment L2 Loss: {alignment_loss_avg:.4f}")
            logger.info(f"Validation Risk Loss: {risk_loss_avg:.4f}")
            logger.info(f"Validation C-index: {c_index_val:.4f}")

            print(
                f"Epoch {epoch} - Val Alignment Loss: {alignment_loss_avg:.4f}, Risk Loss: {risk_loss_avg:.4f}, C-index: {c_index_val:.4f}")

    # Display total training time
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"[INFO] Total time taken to train the model: {elapsed_time:.2f}s")
    logger.info(f"Total time taken to train the model: {elapsed_time:.2f}s")

    # Save the final model
    print("[INFO] Saving final model ...")
    torch.save(model_risk.state_dict(), path_model)

    # Log model as artifact to Weights & Biases
    artifact = wandb.Artifact("model", type="model")
    artifact.add_file(path_model)
    wandb.log_artifact(artifact)

    # Finish WandB run
    wandb.finish()


def train_val_jointly_img_alignment(
    train_loader,
    valid_loader,
    device,
    learning_rate,
    weight_decay,
    num_epochs,
    path_loggger,
    path_model,
    id,
    use_scheduler,
    out_dir,
    accumulation_steps,
    patience_lr_scheduler,
    patience,
    use_img_feat_alignment,
    lr_decay,
    dataset,
):
    print("[INFO] Training the network...")
    start_time = time.time()

    # Initialize logger
    logger = create_logger(path_loggger)
    logger.info(f"Number of Training Epochs: {num_epochs}")

    # Load pre-trained registration model
    path_saved_model = (
        "/storage/CsawCC/NICE-Trans_train_results/model_registration_training_id_125_last_epoch.pth"
        if dataset == "CSAW"
        else "/storage/EMBED/NICE-Trans_train_results_embed/model_registration_training_id_2_last_epoch.pth"
    )
    print("Path reg model:", path_saved_model)

    model_reg = MammoRegNet()
    model_reg.load_state_dict(torch.load(path_saved_model, map_location=device))
    model_reg.to(device).eval()

    # Initialize main risk model
    model_risk = (
        CombinedImgAlignmentRiskModel_downsample_img_deformation_field(num_years=5, registration_model=model_reg)
        if use_img_feat_alignment == "True"
        else CombinedImgAlignmentRiskModel(num_years=5, registration_model=model_reg)
    )
    model_risk.to(device)

    optimizer = torch.optim.Adam(
        model_risk.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    if use_scheduler == "True":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=lr_decay,
            patience=patience_lr_scheduler, verbose=True
        )
        print("Scheduler initialized.")

    # Initialize WandB
    wandb.init(
        project="EMBED_Risk_Prediction",
        config={
            "optimizer": optimizer.__class__.__name__,
            "architecture": "TemporalRiskPrediction",
            "dataset": "EMBED",
            "epochs": num_epochs,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "model": model_risk.__class__.__name__,
        },
    )

    # Define metrics
    wandb.define_metric("epoch", hidden=True)
    for metric in [
        "Training Loss", "Training Accuracy", "Training Alignment Loss", "Training Risk Loss",
        "Validation Validation Risk Loss", "Validation Alignment L2 Loss",
        "Training C-index", "Validation C-index",
        "Year 1 AUC", "Year 2 AUC", "Year 3 AUC", "Year 4 AUC", "Year 5 AUC"
    ]:
        wandb.define_metric(metric, step_metric="epoch")

    # Setup for early stopping and loss
    best_c_index = 0
    patience_counter = 0
    alignment_loss_fn = nn.MSELoss()

    # Training Loop
    for epoch in tqdm(range(num_epochs)):
        print(f"[INFO] Epoch {epoch}")
        logger.info(f"##### Epoch {epoch} #####")

        model_risk.train()
        train_running_risk_loss = 0.0
        predictions, event_times, event_observed = [], [], []
        counter = 0

        for idx, batch in enumerate(train_loader):
            counter += 1
            torch.cuda.empty_cache()

            # Move data to device
            current_image = batch["current_image"].to(device).float()
            prior_image = batch["previous_image"].to(device).float()
            time_gap = batch["time_gap"].to(device)
            target = batch["target"]
            target_prior = batch["target_prior"]
            y_mask = batch["y_mask"]
            y_mask_prior = batch["y_mask_prior"]
            event_times_batch = batch["event_times"].to(device).float()
            event_observed_batch = batch["event_observed"].to(device).float()

            # Forward pass
            outputs = model_risk(current_image, prior_image, time_gap)
            risk_pred = outputs["risk_prediction"]
            del current_image, prior_image

            # Loss computation
            risk_loss_fused = get_risk_loss_BCE(risk_pred["pred_fused"], target, y_mask)
            risk_loss_cur = get_risk_loss_BCE(risk_pred["pred_cur"], target, y_mask)
            risk_loss_pri = get_risk_loss_BCE(risk_pred["pred_pri"], target_prior, y_mask_prior)
            risk_loss = (risk_loss_fused + risk_loss_cur + risk_loss_pri) / accumulation_steps

            train_running_risk_loss += risk_loss.item() * accumulation_steps

            # Backward + Optimization with Gradient Accumulation
            risk_loss.backward()
            if (idx + 1) % accumulation_steps == 0 or (idx + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()

            # Track metrics
            predictions.append(risk_pred["pred_fused"].detach().cpu().numpy())
            event_times.append(event_times_batch.cpu().numpy())
            event_observed.append(event_observed_batch.cpu().numpy())

        # Epoch metrics
        predictions = np.concatenate(predictions, axis=0)
        event_times = np.concatenate(event_times, axis=0)
        event_observed = np.concatenate(event_observed, axis=0)

        censoring_dist = get_censoring_dist(event_times, event_observed)
        c_index = concordance_index_ipcw(event_times, predictions, event_observed, censoring_dist)
        avg_train_loss = train_running_risk_loss / counter

        # Log to WandB
        wandb.log({
            "Training Risk Loss": avg_train_loss,
            "Training C-index": c_index,
            "epoch": epoch
        })

        print(f"Epoch {epoch} | Risk Loss: {avg_train_loss:.4f} | C-index: {c_index:.4f}")

        ##################      Validation      ###################

        with torch.inference_mode():
            model_risk.eval()

            val_running_risk_loss = 0.0
            val_running_alignment_loss = 0.0
            predictions, event_times, event_observed = [], [], []
            counter = 0

            for idx, batch_val in enumerate(valid_loader):
                torch.cuda.empty_cache()
                counter += 1

                # Move data to device
                current_image_val = batch_val["current_image"].to(device).float()
                prior_image_val = batch_val["previous_image"].to(device).float()
                time_gap_val = batch_val["time_gap"].to(device)
                event_times_batch = batch_val["event_times"].to(device).float()
                event_observed_batch = batch_val["event_observed"].to(device).float()
                target_val = batch_val["target"]
                target_prior_val = batch_val["target_prior"]
                y_mask_val = batch_val["y_mask"]
                y_mask_val_prior = batch_val["y_mask_prior"]

                # Forward pass
                outputs_val = model_risk(current_image_val, prior_image_val, time_gap_val)
                del current_image_val, prior_image_val

                aligned_prior = outputs_val["aligned_prior_feature"]
                current_features = outputs_val["current_feature"]
                alignment_loss = alignment_loss_fn(aligned_prior, current_features)
                val_running_alignment_loss += alignment_loss.item()

                # Risk prediction losses
                risk_pred = outputs_val["risk_prediction"]
                loss_fused = get_risk_loss_BCE(risk_pred["pred_fused"], target_val, y_mask_val)
                loss_cur = get_risk_loss_BCE(risk_pred["pred_cur"], target_val, y_mask_val)
                loss_pri = get_risk_loss_BCE(risk_pred["pred_pri"], target_prior_val, y_mask_val_prior)
                total_risk_loss = loss_fused + loss_cur + loss_pri
                val_running_risk_loss += total_risk_loss.item()

                # Collect metrics
                predictions.append(risk_pred["pred_fused"].cpu().numpy())
                event_times.append(event_times_batch.cpu().numpy())
                event_observed.append(event_observed_batch.cpu().numpy())

            # Compute validation metrics
            predictions = np.concatenate(predictions, axis=0)
            event_times = np.concatenate(event_times, axis=0)
            event_observed = np.concatenate(event_observed, axis=0)

            # AUCs
            auc_results = compute_auc_x_year_auc(predictions, event_times, event_observed)
            for year, auc in auc_results.items():
                print(f"Year {year + 1}: AUC = {auc:.4f}")
                wandb.log({f"Year {year + 1} AUC": auc, "epoch": epoch})

            # C-index
            c_index = concordance_index_ipcw(event_times, predictions, event_observed, censoring_dist)

            # Epoch loss averages
            avg_val_risk_loss = val_running_risk_loss / counter
            avg_val_align_loss = val_running_alignment_loss / counter

            # Learning rate scheduling
            if use_scheduler == "True":
                scheduler.step(c_index)

                # Early stopping check
                if c_index > best_c_index:
                    best_c_index = c_index
                    patience_counter = 0
                    best_model_path = os.path.join(out_dir, f"best_model_risk_prediction_id-{id}.pth")
                    torch.save(model_risk.state_dict(), best_model_path)
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        early_stop_path = os.path.join(out_dir, f"early_stopping_risk_prediction_id-{id}.pth")
                        torch.save(model_risk.state_dict(), early_stop_path)
                        print("Early stopping triggered.")
                        break

            # Logging
            wandb.log({
                "Validation Risk Loss": avg_val_risk_loss,
                "Validation Alignment L2 Loss": avg_val_align_loss,
                "Validation C-index": c_index,
                "epoch": epoch
            })

            logger.info(f"Validation Risk Loss: {avg_val_risk_loss:.4f}")
            logger.info(f"Validation Alignment L2 Loss: {avg_val_align_loss:.4f}")
            logger.info(f"Validation C-index: {c_index:.4f}")

            print(f"Epoch {epoch}")
            print(f"Validation Risk Loss: {avg_val_risk_loss:.4f}")
            print(f"Validation Alignment L2 Loss: {avg_val_align_loss:.4f}")
            print(f"Validation C-index: {c_index:.4f}")

    ############## Training Summary & Final Save ##############
    end_time = time.time()
    total_time = end_time - start_time
    print(f"[INFO] Total training time: {total_time:.2f}s")
    logger.info(f"Total training time: {total_time:.2f}s")

    print("[INFO] Saving final model...")
    torch.save(model_risk.state_dict(), path_model)

    # Log model artifact to WandB
    artifact = wandb.Artifact("model", type="model")
    artifact.add_file(path_model)
    wandb.log_artifact(artifact)
    wandb.finish()
