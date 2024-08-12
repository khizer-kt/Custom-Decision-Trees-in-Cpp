#include <iostream>

class Node {
public:
    bool is_leaf;
    int feature_index;
    double threshold;
    double value;
    Node* left;
    Node* right;

    Node() : is_leaf(false), feature_index(-1), threshold(0.0), value(0.0), left(NULL), right(NULL) {}
};

class DecisionTree {
private:
    Node* root;

    double calculate_gini(const int* y, int start, int end) {
        int count[2] = {0, 0};
        for (int i = start; i <= end; ++i) {
            count[y[i]]++;
        }
        double p0 = static_cast<double>(count[0]) / (end - start + 1);
        double p1 = static_cast<double>(count[1]) / (end - start + 1);
        return 1.0 - (p0 * p0 + p1 * p1);
    }

    double calculate_information_gain(int* X, const int* y, int start, int end, int feature_index, double threshold, int num_features) {
        int left_count = 0;
        int right_count = 0;
        for (int i = start; i <= end; ++i) {
            if (X[i * num_features + feature_index] <= threshold) {
                left_count++;
            } else {
                right_count++;
            }
        }

        int left_y[left_count], right_y[right_count];
        int left_index = 0, right_index = 0;

        for (int i = start; i <= end; ++i) {
            if (X[i * num_features + feature_index] <= threshold) {
                left_y[left_index++] = y[i];
            } else {
                right_y[right_index++] = y[i];
            }
        }

        double left_gini = calculate_gini(left_y, 0, left_count - 1);
        double right_gini = calculate_gini(right_y, 0, right_count - 1);

        double p_left = static_cast<double>(left_count) / (end - start + 1);
        double p_right = static_cast<double>(right_count) / (end - start + 1);

        return calculate_gini(y, start, end) - (p_left * left_gini + p_right * right_gini);
    }

    void best_split(int* X, const int* y, int start, int end, int& best_feature_index, double& best_threshold, double& best_gain, int num_features) {
        best_gain = -1.0;
        best_feature_index = -1;
        best_threshold = 0.0;

        for (int feature_index = 0; feature_index < num_features; ++feature_index) {
            for (int i = start; i <= end; ++i) {
                double threshold = X[i * num_features + feature_index];
                double gain = calculate_information_gain(X, y, start, end, feature_index, threshold, num_features);
                if (gain > best_gain) {
                    best_gain = gain;
                    best_feature_index = feature_index;
                    best_threshold = threshold;
                }
            }
        }
    }

    Node* build_tree(int* X, const int* y, int start, int end, int depth, int num_features) {
        Node* node = new Node();

        int count[2] = {0, 0};
        for (int i = start; i <= end; ++i) {
            count[y[i]]++;
        }

        if (depth == 0 || count[0] == 0 || count[1] == 0) {
            node->is_leaf = true;
            node->value = (count[0] > count[1]) ? 0 : 1;
            return node;
        }

        int best_feature_index;
        double best_threshold;
        double best_gain;
        best_split(X, y, start, end, best_feature_index, best_threshold, best_gain, num_features);

        if (best_gain == 0.0) {
            node->is_leaf = true;
            node->value = (count[0] > count[1]) ? 0 : 1;
            return node;
        }

        node->feature_index = best_feature_index;
        node->threshold = best_threshold;

        int left_count = 0;
        for (int i = start; i <= end; ++i) {
            if (X[i * num_features + best_feature_index] <= best_threshold) {
                left_count++;
            }
        }

        int right_count = end - start + 1 - left_count;
        int left_X[left_count * num_features], left_y[left_count];
        int right_X[right_count * num_features], right_y[right_count];
        int left_index = 0, right_index = 0;

        for (int i = start; i <= end; ++i) {
            if (X[i * num_features + best_feature_index] <= best_threshold) {
                for (int j = 0; j < num_features; ++j) {
                    left_X[left_index * num_features + j] = X[i * num_features + j];
                }
                left_y[left_index++] = y[i];
            } else {
                for (int j = 0; j < num_features; ++j) {
                    right_X[right_index * num_features + j] = X[i * num_features + j];
                }
                right_y[right_index++] = y[i];
            }
        }

        node->left = build_tree(left_X, left_y, 0, left_count - 1, depth - 1, num_features);
        node->right = build_tree(right_X, right_y, 0, right_count - 1, depth - 1, num_features);

        return node;
    }

    double predict_single(Node* node, int* X, int num_features) {
        if (node->is_leaf) {
            return node->value;
        }
        if (X[node->feature_index] <= node->threshold) {
            return predict_single(node->left, X, num_features);
        } else {
            return predict_single(node->right, X, num_features);
        }
    }

public:
    DecisionTree() : root(NULL) {}

    void fit(int* X, int* y, int num_samples, int num_features, int max_depth) {
        root = build_tree(X, y, 0, num_samples - 1, max_depth, num_features);
    }

    double predict(int* X, int num_features) {
        return predict_single(root, X, num_features);
    }
};

int main() {
    // Dataset with 3 features: age, likes dogs, likes gravity, and target label: going to be an astronaut
    int X[] = {
        24, 0, 0,
        30, 1, 1,
        36, 0, 1,
        36, 0, 0,
        42, 0, 0,
        44, 1, 1,
        46, 1, 0,
        47, 1, 1,
        47, 0, 1,
        51, 1, 1
    };
    int y[] = {0, 1, 1, 0, 0, 1, 0, 1, 0, 1};

    int num_samples = 10;
    int num_features = 3;

    DecisionTree tree;
    tree.fit(X, y, num_samples, num_features, 3);

    // Test prediction for a new person with age 40, likes dogs (1), likes gravity (1)
    int test_point[] = {40, 1, 1};
    std::cout << "Prediction: " << tree.predict(test_point, num_features) << std::endl;

    return 0;
}
