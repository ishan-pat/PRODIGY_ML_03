import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report

class MLApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ML Model Trainer")
        self.root.geometry("800x600")
        self.root.configure(bg="#e0f7fa")

        # Frame for training dataset selection
        self.frame1 = tk.Frame(root, bg="#e0f7fa")
        self.frame1.pack(pady=10, padx=20, fill=tk.X)

        self.label1 = tk.Label(self.frame1, text="Select Training Dataset:", font=("Helvetica", 12), bg="#e0f7fa")
        self.label1.pack(side=tk.LEFT, padx=5)

        self.button1 = tk.Button(self.frame1, text="Browse", command=self.load_train_dataset, bg="#4db6ac", fg="white", font=("Helvetica", 10, "bold"))
        self.button1.pack(side=tk.LEFT, padx=5)

        self.train_dataset_path = tk.Entry(self.frame1, width=50, font=("Helvetica", 10))
        self.train_dataset_path.pack(side=tk.LEFT, padx=5)

        # Frame for testing dataset selection
        self.frame2 = tk.Frame(root, bg="#e0f7fa")
        self.frame2.pack(pady=10, padx=20, fill=tk.X)

        self.label2 = tk.Label(self.frame2, text="Select Testing Dataset:", font=("Helvetica", 12), bg="#e0f7fa")
        self.label2.pack(side=tk.LEFT, padx=5)

        self.button2 = tk.Button(self.frame2, text="Browse", command=self.load_test_dataset, bg="#4db6ac", fg="white", font=("Helvetica", 10, "bold"))
        self.button2.pack(side=tk.LEFT, padx=5)

        self.test_dataset_path = tk.Entry(self.frame2, width=50, font=("Helvetica", 10))
        self.test_dataset_path.pack(side=tk.LEFT, padx=5)

        # Frame for model selection
        self.frame3 = tk.Frame(root, bg="#e0f7fa")
        self.frame3.pack(pady=10, padx=20, fill=tk.X)

        self.label3 = tk.Label(self.frame3, text="Select Model:", font=("Helvetica", 12), bg="#e0f7fa")
        self.label3.pack(side=tk.LEFT, padx=5)

        self.model_var = tk.StringVar()
        self.model_combo = ttk.Combobox(self.frame3, textvariable=self.model_var, font=("Helvetica", 10))
        self.model_combo['values'] = ("SVM Linear", "SVM RBF", "Decision Tree")
        self.model_combo.current(0)
        self.model_combo.pack(side=tk.LEFT, padx=5)

        # Frame for buttons
        self.frame4 = tk.Frame(root, bg="#e0f7fa")
        self.frame4.pack(pady=10, padx=20, fill=tk.X)

        self.train_button = tk.Button(self.frame4, text="Train Model", command=self.train_model, bg="#4db6ac", fg="white", font=("Helvetica", 10, "bold"))
        self.train_button.pack(side=tk.LEFT, padx=5)

        self.result_button = tk.Button(self.frame4, text="Show Results", command=self.show_results, bg="#4db6ac", fg="white", font=("Helvetica", 10, "bold"))
        self.result_button.pack(side=tk.LEFT, padx=5)

        self.results_text = tk.Text(root, height=15, width=80, font=("Courier", 10), wrap=tk.WORD, bg="#ffffff", fg="#000000", borderwidth=2, relief="sunken")
        self.results_text.pack(pady=10, padx=20, fill=tk.BOTH, expand=True)
        
        self.results_text.tag_configure("header", font=("Helvetica", 12, "bold"), foreground="#4caf50")
        self.results_text.tag_configure("label", font=("Helvetica", 10, "bold"), foreground="#ff5722")
        self.results_text.tag_configure("data", font=("Courier", 10), foreground="#000000")

        self.train_images = None
        self.train_labels = None
        self.test_images = None
        self.test_labels = None
        self.test_filenames = None
        self.clf = None

    def load_train_dataset(self):
        folder_selected = filedialog.askdirectory()
        self.train_dataset_path.delete(0, tk.END)
        self.train_dataset_path.insert(0, folder_selected)
        self.train_images, self.train_labels = self.load_images_from_folder(folder_selected)
        messagebox.showinfo("Dataset", "Training Dataset Loaded Successfully!")

    def load_test_dataset(self):
        folder_selected = filedialog.askdirectory()
        self.test_dataset_path.delete(0, tk.END)
        self.test_dataset_path.insert(0, folder_selected)
        self.test_images, self.test_labels, self.test_filenames = self.load_images_from_folder(folder_selected, include_filenames=True)
        messagebox.showinfo("Dataset", "Testing Dataset Loaded Successfully!")

    def load_images_from_folder(self, folder, include_filenames=False):
        images = []
        labels = []
        filenames = []
        for filename in os.listdir(folder):
            img = cv2.imread(os.path.join(folder, filename))
            if img is not None:
                img = cv2.resize(img, (64, 64))  # Resize to 64x64 pixels
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
                img = img.flatten()  # Flatten the image
                images.append(img)
                label = 1 if 'dog' in filename else 0
                labels.append(label)
                if include_filenames:
                    filenames.append(filename)
        if include_filenames:
            return np.array(images), np.array(labels), filenames
        return np.array(images), np.array(labels)

    def train_model(self):
        if self.train_images is None or self.train_labels is None:
            messagebox.showerror("Error", "No training dataset loaded.")
            return

        # Scale the images
        scaler = StandardScaler()
        images_scaled = scaler.fit_transform(self.train_images)

        # Reduce dimensionality with PCA
        n_samples, n_features = images_scaled.shape
        n_components = min(n_samples, n_features, 20)  # Adjust n_components based on dataset size
        pca = PCA(n_components=n_components)
        images_pca = pca.fit_transform(images_scaled)

        # Split the dataset
        X_train, X_test, y_train, y_test = train_test_split(images_pca, self.train_labels, test_size=0.2, random_state=42)

        model_choice = self.model_var.get()
        if model_choice == "SVM Linear":
            self.clf = svm.SVC(kernel='linear')
        elif model_choice == "SVM RBF":
            self.clf = svm.SVC(kernel='rbf')
        elif model_choice == "Decision Tree":
            from sklearn.tree import DecisionTreeClassifier
            self.clf = DecisionTreeClassifier()

        self.clf.fit(X_train, y_train)
        self.X_test = X_test
        self.y_test = y_test
        messagebox.showinfo("Training", "Model Trained Successfully!")

    def show_results(self):
        if self.clf is None or self.test_images is None or self.test_labels is None:
            messagebox.showerror("Error", "No model trained or testing dataset unavailable.")
            return

        # Scale and transform the test images using the same scaler and PCA
        scaler = StandardScaler()
        test_images_scaled = scaler.fit_transform(self.test_images)
        
        n_samples, n_features = test_images_scaled.shape
        n_components = min(n_samples, n_features, 20)
        pca = PCA(n_components=n_components)
        test_images_pca = pca.fit_transform(test_images_scaled)

        y_pred = self.clf.predict(test_images_pca)
        report = classification_report(self.test_labels, y_pred, output_dict=True, zero_division=1)
        self.results_text.delete(1.0, tk.END)
        
        # Insert styled text
        self.results_text.insert(tk.END, "Classification Report\n", "header")
        
        for label, metrics in report.items():
            if isinstance(metrics, dict):
                self.results_text.insert(tk.END, f"\n{label}\n", "label")
                for metric, value in metrics.items():
                    self.results_text.insert(tk.END, f"{metric: <20} : {value:.2f}\n", "data")
            else:
                self.results_text.insert(tk.END, f"\n{label: <20} : {metrics:.2f}\n", "data")

        # Generate the Excel file with predictions
        df = pd.DataFrame({'Filename': self.test_filenames, 'Actual': self.test_labels, 'Predicted': y_pred})
        df['Class'] = df['Predicted'].apply(lambda x: 'Class 1' if x == 1 else 'Class 2')
        df.to_excel('classification_results.xlsx', index=False)
        messagebox.showinfo("Excel File", "Classification results saved to 'classification_results.xlsx'.")

        
if __name__ == "__main__":
    root = tk.Tk()
    app = MLApp(root)
    root.mainloop()