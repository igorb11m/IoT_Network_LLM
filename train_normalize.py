import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

INPUT_FILE = "iot_features.csv"
OUTPUT_IMAGE = "matrix.png"


def get_category(label):

    label = label.lower()

    if 'cam' in label or 'monitor' in label or 'camera' in label:
        return 'Kamera'


    if 'echo' in label or 'google' in label or 'aria' in label or 'hub' in label or 'bridge' in label:
        return 'Asystent/Hub'


    if 'bulb' in label or 'lifx' in label or 'wemo' in label or 'plug' in label or 'switch' in label or 'hue' in label:
        return 'Gniazdko/Oświetlenie'


    if 'sensor' in label or 'protect' in label or 'nest' in label or 'weather' in label or 'netatmo' in label:
        return 'Sensor'


    return 'Inne'


def main():
    print("1. Loading data...")
    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print(f"Error: Not found {INPUT_FILE}.")
        return


    print("2. Grouping into categories ...")

    df['Category'] = df['Label'].apply(get_category)

    print(f"   Liczba próbek: {len(df)}")
    print(f"   Znalezione kategorie: {df['Category'].unique()}")


    X = df.drop(columns=['Label', 'Category'])

    y = df['Category']


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    print("3. Training on categories ...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)


    print("4. Evaluating ...")
    y_pred = rf_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n=== SCORES FOR CATEGORY ===")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("\Detailed raport:")
    print(classification_report(y_test, y_pred))


    print(f"5. Generating plot {OUTPUT_IMAGE}...")


    cm = confusion_matrix(y_test, y_pred, normalize='true')
    labels = sorted(df['Category'].unique())

    plt.figure(figsize=(10, 8))


    sns.heatmap(cm, annot=True, fmt='.1%', cmap='Greens',
                xticklabels=labels, yticklabels=labels,
                annot_kws={"size": 12})

    plt.title(f'Normalized error matrix (Accuracy: {accuracy * 100:.1f}%)', fontsize=14)
    plt.ylabel('True Category', fontsize=12)
    plt.xlabel('Predicted Category', fontsize=12)

    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE)
    print(f"Saved plot as {OUTPUT_IMAGE}")


if __name__ == "__main__":
    main()