import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

u1 = np.array([-1, 1]).T
u2 = np.array([-2.5, 2.5]).T
u3 = np.array([-4.5, 4.5]).T
sigma1 = np.eye(2)
sigma2 = np.eye(2)
sigma3 = np.eye(2)


def create_data(num_train = 700, num_test = 300):
    synthetic_samples = []
    synthetic_test_samples = []
    actual_test_labels = []
    for i in range(num_test + num_train):
        label = np.random.choice([1, 2, 3], p=[1/3, 1/3, 1/3])
        if label == 1:
            sample = np.random.multivariate_normal(u1, sigma1)
        elif label == 2:
            sample = np.random.multivariate_normal(u2, sigma2)
        else:
            sample = np.random.multivariate_normal(u3, sigma3)
        if i < num_train:
            synthetic_samples.append((sample, label))
        else:
            actual_test_labels.append(label)
            synthetic_test_samples.append(sample)

    sample_data = np.array([i for i, _ in synthetic_samples])
    sample_labels = np.array([l for _, l in synthetic_samples])
    synthetic_test_samples = np.array(synthetic_test_samples)
    return sample_data, sample_labels, synthetic_test_samples, actual_test_labels

def knn_model_run(i=1, sample_data=None, sample_labels=None, synthetic_test_samples=None, actual_test_labels=None):
    knn_classifier = KNeighborsClassifier(n_neighbors=i, p=2)  # l2 norm
    # training model
    knn_classifier.fit(sample_data, sample_labels)

    # Predict labels for the test data
    predicted_labels = knn_classifier.predict(synthetic_test_samples)

    error_rate_train = 1 - knn_classifier.score(sample_data, sample_labels)

    error_rate_test = (sum(i != y for i, y in zip(predicted_labels, actual_test_labels))) / len(predicted_labels)

    return error_rate_train, error_rate_test

def Q1_p2_3(sample_data, sample_labels, synthetic_test_samples):
    colors = ['r', 'g', 'b']
    labels = ['Gaussian 1', 'Gaussian 2', 'Gaussian 3']

    plt.figure(figsize=(10, 8))
    label_color_map = ['r', 'b', 'g']
    # Plot the data points with their corresponding labels
    for data, label in zip(sample_data, sample_labels):
        plt.scatter(data[0], data[1], color=label_color_map[label - 1])

    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title('Scatter Plot of Samples from Three Gaussian Distributions')
    plt.legend()
    plt.grid(True)
    plt.show()

    # test samples plot
    plt.figure(figsize=(10, 8))
    for data in synthetic_test_samples:
        plt.scatter(data[0], data[1], color='b')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title('Scatter Plot of Test Samples from Three Gaussian Distributions')
    plt.legend()
    plt.grid(True)
    plt.show()

def Q1_p4_5(sample_data, sample_labels, synthetic_test_samples, actual_test_labels):
    for i in range(1, 21):
        # Q1.4,5
        error_rate_train, error_rate_test = knn_model_run(i, sample_data, sample_labels, synthetic_test_samples, actual_test_labels)
        print(f"\nerror rate on knn model with {i} neighbours on train set: {error_rate_train}")
        print(f"error rate on knn model with {i} neighbours on test set: {error_rate_test}")

def Q1_p6():
    m_test = 100
    train_errors = []
    test_errors = []
    m_train_vals = [i for i in range(10, 45, 5)]
    for m_train_i in m_train_vals:
        sample_data, sample_labels, synthetic_test_samples, actual_test_labels = create_data(m_train_i, m_test)
        error_rate_train, error_rate_test = knn_model_run(10, sample_data, sample_labels, synthetic_test_samples, actual_test_labels)
        train_errors.append(error_rate_train)
        test_errors.append(error_rate_test)

    # Plotting the results
    plt.figure(figsize=(10, 6))
    plt.plot(m_train_vals, train_errors, label='Train Error')
    plt.plot(m_train_vals, test_errors, label='Test Error')
    plt.xlabel('Training Set Size (m_train)')
    plt.ylabel('Error Rate')
    plt.title('Train and Test Errors as a function of Training Set Size')
    plt.legend()
    plt.grid(True)
    plt.show()


sample_data, sample_labels, synthetic_test_samples, actual_test_labels = create_data()

#Q1_p2_3(sample_data, sample_labels, synthetic_test_samples)
Q1_p4_5(sample_data, sample_labels, synthetic_test_samples, actual_test_labels)
for _ in range(5):
    Q1_p6()
