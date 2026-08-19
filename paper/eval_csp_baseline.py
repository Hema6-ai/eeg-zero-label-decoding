"""Diagnostic 3: Common Spatial Patterns (CSP) + Logistic Regression, the classical purpose-built
spatial filter for motor imagery, on the same per-subject train/test split as the in-distribution
probe and the EEGNet baseline. If CSP separates classes better than the frozen MAE representation
in-distribution, that is clean evidence the learned representation itself is the bottleneck.

    python -m paper.eval_csp_baseline
"""
import numpy as np
from mne.decoding import CSP
from sklearn.linear_model import LogisticRegression

from paper import config
from paper.train_eegnet_baseline import load_subject_sessions
from paper.utils import classification_metrics


def run(subjects=None, n_components: int = 6):
    subjects = subjects or config.EVAL_SUBJECTS

    fold_results = []
    for subject in subjects:
        X_train, y_train, X_test, y_test, label_encoder = load_subject_sessions(subject)

        csp = CSP(n_components=n_components, reg="ledoit_wolf", log=True)
        train_feats = csp.fit_transform(X_train.astype(np.float64), y_train)
        test_feats = csp.transform(X_test.astype(np.float64))

        clf = LogisticRegression(max_iter=1000)
        clf.fit(train_feats, y_train)
        pred = clf.predict(test_feats)

        metrics = classification_metrics(y_test, pred)
        metrics["subject"] = subject
        fold_results.append(metrics)
        print(f"[csp] subject {subject}: acc={metrics['accuracy']:.4f} kappa={metrics['kappa']:.4f}")

    accs = np.array([r["accuracy"] for r in fold_results])
    kappas = np.array([r["kappa"] for r in fold_results])
    print(f"[csp] in-distribution accuracy: {accs.mean():.4f} +/- {accs.std():.4f}  "
          f"kappa: {kappas.mean():.4f} +/- {kappas.std():.4f}")
    return {"accuracy_mean": accs.mean(), "accuracy_std": accs.std(),
            "kappa_mean": kappas.mean(), "kappa_std": kappas.std()}, fold_results


if __name__ == "__main__":
    run()
