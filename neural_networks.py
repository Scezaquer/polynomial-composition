import numpy as np
from math import factorial, comb
from numpy import polynomial as Polynomial
import matplotlib.pyplot as plt
import time
import random
from polynomial_utils import compose, compose_layers, l2_norm, l2_coefficient_norm, plot_polynomials, carleman, carleman_matrix
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.inspection import permutation_importance


def solve_for_h(g, target_poly):
    # Find shift variable d
    h0_poly = g - target_poly.coef[0]
    roots = h0_poly.roots()
    real_roots = [root.real for root in roots]  # if np.isreal(root)]
    if len(real_roots) == 0:
        raise ValueError("No real roots found")

    solutions = []
    for d in real_roots:
        # d = min(real_roots, key=abs).real

        # Shift g and target_poly
        shifted_q = target_poly - d
        shifted_g0 = g(d) - d
        shifted_gl = []
        for l in range(1, g.coef.size):
            shifted_gl.append(
                sum([g.coef[j] * d**(j-l) * comb(j, l) for j in range(l, g.coef.size)]))
        # satisfies g(x+d)-d = shifted_g(x)
        shifted_g = Polynomial.Polynomial([shifted_g0] + shifted_gl)

        # h1 = target_poly.coef[1] / g.coef[1]
        # h2 = (target_poly.coef[2] - g.coef[2] * h1**2) / g.coef[1]
        # h3 = (target_poly.coef[3] - g.coef[3] * h1**3 - g.coef[2] * 2 * h1 * h2) / g.coef[1]

        # turns out this doesn't use q[0] so we can just use target_poly instead of shifted_q
        h1 = target_poly.coef[1] / shifted_g.coef[1]
        h2 = (target_poly.coef[2] - shifted_g.coef[2]
              * h1**2) / shifted_g.coef[1]
        h3 = (target_poly.coef[3] - shifted_g.coef[3] * h1**3 -
              shifted_g.coef[2] * 2 * h1 * h2) / shifted_g.coef[1]
        h = Polynomial.Polynomial([d, h1, h2, h3])
        # print(h.coef)
        solutions.append(h)
    # Is there a better way to choose the best h?
    h = min(solutions, key=lambda h: l2_coefficient_norm(
        compose(g, h), target_poly))
    return h


def carleman_upper_triangular_solver(h, g, target_poly: Polynomial, iteration: int = 10, size: int = 10, w=None, verbose=False):
    """
    Given a target polynomial, find a polynomial that approximates the target
    polynomial using the Carleman matrix up to the nth row and mth column. If m
    is not provided, the Carleman matrix will be square.
    """
    target_carleman = carleman_matrix(target_poly, size, 4)
    if verbose:
        print(f"g: {g}")
        print(f"h: {h}")
    for i in range(iteration):

        # We only need the first 10 columns of the Carleman matrix, one for
        # each coeff of the target polynomial, and the first 4 rows, one for
        # each coeff of the g polynomial we try to find. We transpose the
        # matrix so that the linalg.solve function can solve the system of
        # equations.
        m_h = carleman_matrix(h, 4, 10).T

        # Solve the system of equations to find the coefficients of the g
        # polynomial. We use lstsq since the system is overdetermined

        g = Polynomial.Polynomial(np.linalg.lstsq(
            m_h, target_poly.coef, rcond=None)[0])

        # if verbose:
        #     composed = compose_layers([h, g])
        #     plot_polynomials(composed, target_poly, i+0.5)

        h = solve_for_h(g, target_poly)

        if verbose:
            print(f"g: {g}")
            print(f"h: {h}")

            composed = compose(g, h)
            plot_polynomials(composed, target_poly, i, linspace_range=(0, 1))

    if verbose:
        print()
    return h, g


def new_poly(width):
    p1 = Polynomial.Polynomial(np.random.uniform(-width, width, 4))
    p2 = Polynomial.Polynomial(np.random.uniform(-width, width, 4))
    # p1 = Polynomial.Polynomial.fromroots(np.random.uniform(-width, width, 3))
    # p2 = Polynomial.Polynomial.fromroots(np.random.uniform(-width, width, 3))
    target_poly = compose(p1, p2)
    return p1, p2, target_poly


def create_dataset(num_samples=5000, width=1.5):
    """
    Create a dataset of polynomial compositions and check if the Carleman solver converges.

    Returns:
    - X_coefs: coefficients of p1, p2, target_poly
    - X_roots: roots of p1, p2, target_poly
    - y: 1 if converged, 0 if not
    """
    X_coefs = []
    X_roots = []
    y = []

    for _ in tqdm(range(num_samples), desc="Generating dataset"):
        try:
            # Generate polynomials
            p1, p2, target_poly = new_poly(width)

            # Get coefficients (pad to fixed size)
            p1_coefs = np.pad(p1.coef, (0, 4 - len(p1.coef)))
            p2_coefs = np.pad(p2.coef, (0, 4 - len(p2.coef)))
            target_coefs = np.pad(
                target_poly.coef, (0, 10 - len(target_poly.coef)))

            # Get roots
            p1_roots = p1.roots()
            p2_roots = p2.roots()
            target_roots = target_poly.roots()

            # Pad roots to fixed size and split into real/imag parts
            p1_roots_padded = np.pad(p1_roots, (0, 3 - len(p1_roots)))
            p2_roots_padded = np.pad(p2_roots, (0, 3 - len(p2_roots)))
            target_roots_padded = np.pad(
                target_roots, (0, 9 - len(target_roots)))

            p1_roots_real = np.real(p1_roots_padded)
            p1_roots_imag = np.imag(p1_roots_padded)
            p2_roots_real = np.real(p2_roots_padded)
            p2_roots_imag = np.imag(p2_roots_padded)
            target_roots_real = np.real(target_roots_padded)
            target_roots_imag = np.imag(target_roots_padded)

            # Random initial polynomials
            h0 = Polynomial.Polynomial(np.random.uniform(-0.5, 0.5, 4))
            g0 = Polynomial.Polynomial(np.random.uniform(-0.5, 0.5, 4))

            # Check if algorithm converges
            h, g = carleman_upper_triangular_solver(h0, g0, target_poly, 100)
            composed = compose(g, h)
            error = l2_coefficient_norm(composed, target_poly)
            converged = 1 if error < 1e-6 else 0

            # Store data
            X_coefs.append(np.concatenate([p1_coefs, p2_coefs, target_coefs]))
            X_roots.append(np.concatenate([
                p1_roots_real, p1_roots_imag,
                p2_roots_real, p2_roots_imag,
                target_roots_real, target_roots_imag
            ]))
            y.append(converged)

        except Exception as e:
            continue

    return np.array(X_coefs), np.array(X_roots), np.array(y)


def build_convergence_model():
    """
    Build a neural network model to predict if Carleman solver will converge
    """
    # Input branches
    coef_input = layers.Input(
        shape=(18,), name="coefficients")  # 4+4+10 = 18 coeffs
    # 3+3+9 = 15 roots * 2 (real/imag) = 30
    roots_input = layers.Input(shape=(30,), name="roots")

    # Process coefficients
    x_coef = layers.Dense(64, activation='leaky_relu')(coef_input)
    x_coef = layers.BatchNormalization()(x_coef)
    x_coef = layers.Dense(32, activation='leaky_relu')(x_coef)

    # Process roots
    x_roots = layers.Dense(128, activation='leaky_relu')(roots_input)
    x_roots = layers.BatchNormalization()(x_roots)
    x_roots = layers.Dense(64, activation='leaky_relu')(x_roots)
    x_roots = layers.Dense(32, activation='leaky_relu')(x_roots)

    # Combine features
    combined = layers.concatenate([x_coef, x_roots])

    # Additional layers
    x = layers.Dense(64, activation='leaky_relu')(combined)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(32, activation='leaky_relu')(x)
    x = layers.Dropout(0.2)(x)

    # Output layer
    output = layers.Dense(1, activation='sigmoid',
                          name="convergence_probability")(x)

    # Create model
    model = models.Model(inputs=[coef_input, roots_input], outputs=output)

    # Compile model
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC()]
    )

    return model


def train_convergence_model(X_coefs, X_roots, y, num_samples=5000, width=1.5):

    # Split into train/validation sets
    X_coefs_train, X_coefs_val, X_roots_train, X_roots_val, y_train, y_val = train_test_split(
        X_coefs, X_roots, y, test_size=0.2, random_state=42, stratify=y
    )

    # Build model
    model = build_convergence_model()

    # Train model
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=20, restore_best_weights=True
    )

    history = model.fit(
        [X_coefs_train, X_roots_train], y_train,
        epochs=100,
        batch_size=32,
        validation_data=([X_coefs_val, X_roots_val], y_val),
        callbacks=[early_stopping]
    )

    # Plot training history
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    ax1.plot(history.history['loss'], label='Training Loss')
    ax1.plot(history.history['val_loss'], label='Validation Loss')
    ax1.set_title('Loss')
    ax1.set_xlabel('Epoch')
    ax1.legend()

    ax2.plot(history.history['accuracy'], label='Training Accuracy')
    ax2.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax2.set_title('Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.legend()

    plt.tight_layout()
    plt.show()

    # Evaluate on validation set
    val_loss, val_acc, val_auc = model.evaluate(
        [X_coefs_val, X_roots_val], y_val)
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_acc:.4f}")
    print(f"Validation AUC: {val_auc:.4f}")

    return model, history


def analyze_feature_importance(model, X_coefs, X_roots, y):
    """
    Analyze feature importance using permutation importance.

    Args:
        model: Trained model
        X_coefs: Coefficient features
        X_roots: Root features
        y: Target values
    """

    # Create a wrapper class that implements the scikit-learn estimator interface
    class KerasModelWrapper:
        def __init__(self, keras_model):
            self.keras_model = keras_model

        def fit(self, X, y):
            # Not used but required by scikit-learn
            return self

        def predict(self, X):
            # X is a pandas DataFrame or numpy array, we need to split it into our two input types
            n_coefs = 18
            if isinstance(X, pd.DataFrame):
                coefs = X.iloc[:, :n_coefs].values
                roots = X.iloc[:, n_coefs:].values
            else:  # numpy array
                coefs = X[:, :n_coefs]
                roots = X[:, n_coefs:]
            return self.keras_model.predict([coefs, roots], verbose=0).flatten()

        def score(self, X, y):
            # Required by permutation_importance
            from sklearn.metrics import accuracy_score
            y_pred = (self.predict(X) > 0.5).astype(int)
            return accuracy_score(y, y_pred)

    # Combine inputs into a single array or dataframe
    feature_names = [f'coef_{i}' for i in range(
        18)] + [f'root_{i}' for i in range(30)]
    X_combined = np.concatenate([X_coefs, X_roots], axis=1)

    # Create the wrapper and calculate permutation importance
    model_wrapper = KerasModelWrapper(model)
    perm_importance = permutation_importance(
        model_wrapper, X_combined, y,
        n_repeats=5,
        random_state=42
    )

    # Display results
    print("Feature importance (higher is more important):")
    importances = list(zip(feature_names, perm_importance.importances_mean))
    importances.sort(key=lambda x: x[1], reverse=True)

    # Plot top 20 features
    top_n = 20
    plt.figure(figsize=(12, 8))
    features = [imp[0] for imp in importances[:top_n]]
    scores = [imp[1] for imp in importances[:top_n]]

    plt.barh(range(len(features)), scores, align='center')
    plt.yticks(range(len(features)), features)
    plt.title('Feature Importance (Permutation Importance)')
    plt.xlabel('Importance Score')
    plt.tight_layout()
    plt.show()

    # Return group importance (coefficients vs roots)
    coef_importance = sum([imp[1] for imp in importances if 'coef_' in imp[0]])
    root_importance = sum([imp[1] for imp in importances if 'root_' in imp[0]])
    print(f"\nAggregate importance:")
    print(f"Coefficients: {coef_importance:.4f}")
    print(f"Roots: {root_importance:.4f}")

    return importances


if __name__ == "__main__":
    # Train the model
    num_samples = 10000
    width = 1.5

    # Create dataset
    X_coefs, X_roots, y = create_dataset(num_samples, width)

    # Train the model
    model, history = train_convergence_model(
        X_coefs, X_roots, y, num_samples, width)

    # Analyze feature importance
    # Use only 10% of the data for feature importance analysis to reduce computation time
    sample_size = int(len(y) * 0.10)
    random_indices = np.random.choice(len(y), sample_size, replace=False)
    X_coefs_sample = X_coefs[random_indices]
    X_roots_sample = X_roots[random_indices]
    y_sample = y[random_indices]

    print(
        f"Using {sample_size} samples ({sample_size/len(y):.1%} of data) for feature importance analysis")
    importances = analyze_feature_importance(model, X_coefs, X_roots, y)

    # Save the model
    model.save("carleman_convergence_predictor.h5")
    print("Model saved as 'carleman_convergence_predictor.h5'")
