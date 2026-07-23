\# Real-Time Sign Language to Text Conversion System



A webcam-based AI system that recognizes American Sign Language (ASL) hand signs 

(A-Z) and converts them into text in real time — using only a laptop webcam, 

no special hardware required.



\## How It Works



1\. Webcam captures live video

2\. MediaPipe detects the hand and extracts 21 landmark points (63 numbers)

3\. A Random Forest classifier predicts the sign from those 63 numbers

4\. A stability filter confirms the letter before displaying it



\## Tech Stack



\- Python 3.14

\- MediaPipe (hand landmark detection)

\- OpenCV (webcam + display)

\- scikit-learn (Random Forest classifier)

\- NumPy, Pandas



\## Project Structure



sign\_language\_project/

├── data/               ← landmark CSV (gitignored)

├── model/              ← trained model .pkl (gitignored)

├── collect\_data.py     ← extract landmarks from dataset images

├── train\_model.py      ← train and evaluate the model

├── recognize\_gui.py    ← live webcam recognition app

├── requirements.txt    ← dependencies

└── README.md



\## How To Run



1\. Install dependencies:

&#x20;  pip install -r requirements.txt



2\. Extract landmarks from dataset:

&#x20;  python collect\_data.py



3\. Train the model:

&#x20;  python train\_model.py



4\. Run the live recognition app:

&#x20;  python recognize\_gui.py



\## Controls



\- SPACE     → add a space

\- BACKSPACE → delete last letter

\- ESC       → quit



\## Results



\- Accuracy: (to be updated after training)

\- FPS: (to be updated after testing)



\## Dataset



ASL Alphabet Dataset from Kaggle:

https://www.kaggle.com/datasets/grassknoted/asl-alphabet



\## Future Improvements



\- Support for dynamic signs (J, Z)

\- Multi-hand support

\- Mobile app version

