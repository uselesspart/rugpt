import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QLineEdit, QVBoxLayout, QWidget, QPushButton
from transformers import GPT2LMHeadModel, GPT2Tokenizer


class MyWindow(QMainWindow):
    model_name_or_path: str
    tokenizer: GPT2Tokenizer
    model: GPT2LMHeadModel

    def __init__(self):
        super().__init__()
        model_name_or_path = "sberbank-ai/rugpt3large_based_on_gpt2"
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name_or_path)
        self.model = GPT2LMHeadModel.from_pretrained(model_name_or_path).cuda()
        self.setWindowTitle('Input and Output Window')
        self.setGeometry(100, 100, 400, 200)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout()

        self.input_label = QLabel('Enter text:', self)
        self.layout.addWidget(self.input_label)

        self.input_text = QLineEdit(self)
        self.layout.addWidget(self.input_text)

        self.output_label = QLabel('Output:', self)
        self.layout.addWidget(self.output_label)

        self.output_text = QLabel('', self)
        self.layout.addWidget(self.output_text)

        self.submit_button = QPushButton('Submit', self)
        self.submit_button.clicked.connect(self.show_output)
        self.layout.addWidget(self.submit_button)

        self.central_widget.setLayout(self.layout)

    def show_output(self):
        input_text = self.input_text.text()
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt").cuda()
        out = self.model.generate(input_ids.cuda(), do_sample=True, max_length=50, repetition_penalty=5.0,
                                  top_k=5, top_p=0.95, temperature=1,
                                  num_beams=10,
                                  no_repeat_ngram_size=3)
        generated_text = list(map(self.tokenizer.decode, out))[0]
        self.output_text.setText(generated_text)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MyWindow()
    window.show()
    sys.exit(app.exec_())
