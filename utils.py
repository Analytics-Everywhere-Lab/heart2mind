from sklearn.model_selection import KFold
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.callbacks import Callback


class PrintMetricsCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        print(
            f"Epoch {epoch+1}: "
            f"loss={logs.get('loss'):.4f}, "
            f"accuracy={logs.get('accuracy'):.4f}, "
            f"precision={logs.get('precision'):.4f}, "
            f"recall={logs.get('recall'):.4f}, "
            f"f1_score={logs.get('f1_score'):.4f}, "
            f"auc={logs.get('auc'):.4f} | "
            f"val_loss={logs.get('val_loss'):.4f}, "
            f"val_accuracy={logs.get('val_accuracy'):.4f}, "
            f"val_precision={logs.get('val_precision'):.4f}, "
            f"val_recall={logs.get('val_recall'):.4f}, "
            f"val_f1_score={logs.get('val_f1_score'):.4f}, "
            f"val_auc={logs.get('val_auc'):.4f}"
        )


def get_callbacks(patience=20):
    early_stopping = EarlyStopping(
        monitor="val_auc", patience=patience, mode="max", restore_best_weights=True
    )
    reduce_lr = ReduceLROnPlateau(
        monitor="val_auc", factor=0.5, patience=5, mode="max", min_lr=1e-6, verbose=1,
    )
    return [early_stopping, reduce_lr]


def get_kfold(n_splits=5, random_state=41):
    return KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
