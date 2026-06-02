import os
import torch
from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
import torchvision
from datetime import datetime
from torchvision.ops import box_iou
from collections import defaultdict

# input
# train_loader
# val_loader
# test_loader

#NUM_CLASSES = 2
#num_epochs = 10
#Save_The_Whole_Model = False
#Save_Models_architecture = True


# =========================================================
# =============== FUNKCJE POMOCNICZE ======================
# =========================================================

def run_one_epoch_train(model, loader, optimizer, device, epoch, total_epochs):
    model.train()
    running_loss = 0.0

    batch_size = loader.batch_size
    dataset_size = len(loader.dataset)

    for batch_idx, (images, targets) in enumerate(loader):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        loss = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        first_img = batch_idx * batch_size + 1
        last_img = min(first_img + len(images) - 1, dataset_size)

        print(
            f"Epoch [{epoch}/{total_epochs}] | "
            f"Batch {batch_idx + 1}/{len(loader)} | "
            f"Images {first_img}-{last_img}/{dataset_size} | "
            f"Loss: {loss.item():.4f}"
        )

    return running_loss / max(1, len(loader))


def run_one_epoch_val_loss(model, loader, device):
    model.train()  # wymagane przez torchvision
    running_loss = 0.0

    for images, targets in loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        loss = sum(loss for loss in loss_dict.values())
        running_loss += loss.item()

    return running_loss / max(1, len(loader))


def compute_metrics(model, loader, device, iou_threshold=0.5, score_threshold=0.5):
    model.eval()

    TP = 0
    FP = 0
    FN = 0

    for images, targets in loader:
        images = [img.to(device) for img in images]

        with torch.no_grad():
            outputs = model(images)

        for pred, target in zip(outputs, targets):

            pred_boxes = pred["boxes"].cpu()
            pred_scores = pred["scores"].cpu()
            gt_boxes = target["boxes"].cpu()

            keep = pred_scores >= score_threshold
            pred_boxes = pred_boxes[keep]

            matched_gt = set()

            tp = 0
            fp = 0

            for pb in pred_boxes:

                best_iou = 0
                best_idx = -1

                for i, gt in enumerate(gt_boxes):

                    if i in matched_gt:
                        continue

                    ix1 = max(pb[0], gt[0])
                    iy1 = max(pb[1], gt[1])
                    ix2 = min(pb[2], gt[2])
                    iy2 = min(pb[3], gt[3])

                    inter_area = max(0, ix2 - ix1) * max(0, iy2 - iy1)

                    pb_area = (pb[2] - pb[0]) * (pb[3] - pb[1])
                    gt_area = (gt[2] - gt[0]) * (gt[3] - gt[1])

                    union = pb_area + gt_area - inter_area
                    iou = inter_area / (union + 1e-6)

                    if iou > best_iou:
                        best_iou = iou
                        best_idx = i

                if best_iou >= iou_threshold:
                    tp += 1
                    matched_gt.add(best_idx)
                else:
                    fp += 1

            fn = len(gt_boxes) - len(matched_gt)

            TP += tp
            FP += fp
            FN += fn

    precision = TP / (TP + FP + 1e-6)
    recall = TP / (TP + FN + 1e-6)

    return precision, recall


def evaluate_counting_mae(model, loader, device, score_threshold=0.5):
    model.eval()

    absolute_errors = []
    gt_counts = []
    pred_counts = []

    for images, targets in loader:

        images = [img.to(device) for img in images]
        predictions = model(images)

        for pred, target in zip(predictions, targets):

            gt_count = len(target["boxes"])

            keep = pred["scores"] > score_threshold
            pred_boxes = pred["boxes"][keep]
            pred_count = len(pred_boxes)

            gt_counts.append(gt_count)
            pred_counts.append(pred_count)

            absolute_errors.append(abs(gt_count - pred_count))

    mae = sum(absolute_errors) / max(1, len(absolute_errors))

    print("\nPrzyklad liczenia obiektow:")
    for i in range(min(10, len(gt_counts))):
        print(f"Obraz {i + 1}: GT={gt_counts[i]} | PRED={pred_counts[i]}")

    return mae


# =========================================================
# ================== TRENING ==============================
# =========================================================

def Model_Train(train_loader, val_loader, test_loader, NUM_CLASSES,
                num_epochs=4, Save_The_Whole_Model=True,
                Save_Models_architecture=False, Save_Checkpoint=False):

    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(weights=weights)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        in_features, NUM_CLASSES
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.005,
        momentum=0.9,
        weight_decay=0.0005
    )

    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=5, gamma=0.1
    )

    print("\nRozpoczynam uczenie:\n")

    best_val = float("inf")
    best_state = None

    patience = 3
    no_improve_epochs = 0

    for epoch in range(num_epochs):

        train_loss = run_one_epoch_train(
            model, train_loader, optimizer, device, epoch + 1, num_epochs
        )

        val_loss = run_one_epoch_val_loss(model, val_loader, device)

        lr_scheduler.step()

        print(f"Epoch {epoch+1}/{num_epochs} | train loss: {train_loss:.4f} | val loss: {val_loss:.4f}")

        if no_improve_epochs >= patience:
            print("\nEARLY STOPPING: brak poprawy val loss")
            break

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1

    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss = run_one_epoch_val_loss(model, test_loader, device)
    print(f"\nTest loss (best-val model): {test_loss:.4f}")

    precision, recall = compute_metrics(model, test_loader, device)

    print("\n==================== METRYKI ====================")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print("=================================================\n")

    counting_mae = evaluate_counting_mae(model, test_loader, device)

    print(f"\nMAE liczenia obiektow: {counting_mae:.2f}")

    while True:

        answer = input(
            "\nCzy chcesz zapisac model?\nWpisz: Tak/Nie: "
        ).strip().lower()

        if answer in ("tak", "t"):

            timestamp = datetime.now().strftime("%Y%m%d_%H%M")

            model_path = os.path.join(
                f"fasterrcnn_model_{timestamp}.pth"
            )

            torch.save(model.state_dict(), model_path)

            print(f"\nModel zapisany w: {model_path}")
            break

        elif answer in ("nie", "n"):
            print("\nModel NIE został zapisany.")
            break

        else:
            print("Nieprawidłowa odpowiedz. Wpisz Tak lub Nie.")

    if Save_Checkpoint == True:
        checkpoint_path = os.path.join("fasterrcnn_model_checkpoint.pth")

        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "num_classes": NUM_CLASSES,
            "epoch": num_epochs,
        }, checkpoint_path)

        print(f"Checkpoint zapisany w: {checkpoint_path}")

    return model, device