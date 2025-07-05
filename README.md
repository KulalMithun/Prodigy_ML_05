<h1>🍱 Food Recognition and Calorie Estimation</h1>

<p>This is the <strong>final task (#5)</strong> of my <strong>Prodigy InfoTech Internship</strong> project series. The goal of this project is to recognize food items from images using a deep learning model and estimate their <strong>calorie content</strong>. It uses the popular <a href="https://www.kaggle.com/datasets/kmader/food41" target="_blank">Food-101 dataset</a>, and the calorie data is mapped from a custom <code>calories.csv</code> file.</p>

<hr>

<h2>📌 Project Overview</h2>

<ul>
  <li>🔍 <strong>Objective</strong>: Detect food from images and estimate the calorie value.</li>
  <li>🤖 <strong>Model</strong>: Convolutional Neural Network (CNN) built with TensorFlow/Keras.</li>
  <li>🖼️ <strong>Input</strong>: Food images resized to 128x128.</li>
  <li>📊 <strong>Output</strong>: Predicted food class with corresponding calorie value.</li>
  <li>📈 <strong>Accuracy</strong>: Achieved ~54.16% validation accuracy with only <strong>3 epochs</strong>.</li>
  <li>🧪 <strong>Status</strong>: Proof-of-concept (can be improved with longer training and tuning).</li>
</ul>

<hr>

<h2>🛠️ Tech Stack</h2>

<table>
  <tr>
    <th>Component</th>
    <th>Used In</th>
  </tr>
  <tr>
    <td><strong>Python</strong></td>
    <td>General Programming</td>
  </tr>
  <tr>
    <td><strong>TensorFlow</strong></td>
    <td>CNN model building & training</td>
  </tr>
  <tr>
    <td><strong>Keras</strong></td>
    <td>Layers and model structure</td>
  </tr>
  <tr>
    <td><strong>Matplotlib</strong></td>
    <td>Data visualization (accuracy/loss graphs)</td>
  </tr>
  <tr>
    <td><strong>NumPy</strong></td>
    <td>Image and array handling</td>
  </tr>
  <tr>
    <td><strong>Pandas</strong></td>
    <td>Calorie data loading</td>
  </tr>
  <tr>
    <td><strong>Food-101</strong></td>
    <td>Dataset for food images</td>
  </tr>
</table>

<hr>

<h2>📁 Folder Structure</h2>

<pre>
.
├── food-101/                   # Image dataset (subset used)
│   └── images/
├── calories.csv                # Contains food item and calorie info
├── best_model.h5               # Saved trained model
├── Food.ipynb                  # Main notebook (this repo)
</pre>

<hr>

<h2>🧠 Model Architecture</h2>

<ul>
  <li>Input: 128x128x3 images</li>
  <li>2 Convolutional Layers + MaxPooling</li>
  <li>Flatten Layer</li>
  <li>Dense Layers with Softmax output</li>
  <li>Loss Function: Categorical Crossentropy</li>
  <li>Optimizer: Adam</li>
</ul>

<hr>

<h2>📉 Results</h2>

<table>
  <tr><th>Metric</th><th>Value</th></tr>
  <tr><td>Epochs</td><td>3</td></tr>
  <tr><td>Accuracy</td><td>54.16%</td></tr>
  <tr><td>Image Size</td><td>128 x 128 px</td></tr>
  <tr><td>Optimizer</td><td>Adam</td></tr>
</table>

<p><strong>⚠️ Note:</strong> Only trained for 3 epochs for prototyping. Accuracy can improve with more training and augmentation.</p>

<hr>

<h2>📷 Sample Output</h2>

<p>The model predicts the food item from an input image and displays the calorie count from the dataset.</p>

<pre><code>
Predicted: samosa
Calories: 308 per 100g
</code></pre>


<hr>

<h2>🔍 How to Run</h2>

<ol>
  <li>Clone the repo or open the notebook in Jupyter or Kaggle.</li>
  <li>Ensure <code>food-101</code> dataset and <code>calories.csv</code> are available in the correct paths.</li>
  <li>Run all notebook cells.</li>
  <li>Change <code>img_path</code> in the prediction cell to test different images.</li>
</ol>

<hr>

<h2>📈 Future Improvements</h2>

<ul>
  <li>Train for more epochs (e.g., 20+)</li>
  <li>Use data augmentation to reduce overfitting</li>
  <li>Try transfer learning (MobileNet, ResNet, EfficientNet)</li>
  <li>Use regression model to estimate calories instead of lookup</li>
  <li>Deploy as a web or mobile app</li>
</ul>

<hr>

<h2>🙋‍♂️ About Me</h2>

<p><strong>Mithun</strong><br>
💻 Python & AI Enthusiast<br>
<a href="https://github.com/KulalMithun" target="_blank">🌐 GitHub</a></p>

<hr>

<h2>🎓 Internship</h2>

<p>This project is submitted as <strong>Task 5: Final Project</strong> under the <strong>Prodigy InfoTech Virtual Internship Program</strong> in the AI/ML domain.</p>
