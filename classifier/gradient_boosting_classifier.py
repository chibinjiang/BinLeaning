import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score

from data_set.data_loader import data_loader


def main():
    dataset = data_loader.randomize_classifier_data()
    # 10折的CV, 3次repeat
    classifier = GradientBoostingClassifier()
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    n_scores = cross_val_score(classifier, dataset.trainX, dataset.trainY, scoring='accuracy', cv=cv, n_jobs=1)
    classifier.fit(dataset.trainX, dataset.trainY)
    pred_y = classifier.predict(dataset.trainX)
    print(f"Accuracy: {sum(pred_y == dataset.trainY) / dataset.trainY.shape[0]}")
    print('Mean Accuracy: %.3f (%.3f)' % (np.mean(n_scores), np.std(n_scores)))
    import ipdb; ipdb.set_trace()


if __name__ == '__main__':
    """
    python -m classifier.gradient_boosting_classifier
    """
    main()
