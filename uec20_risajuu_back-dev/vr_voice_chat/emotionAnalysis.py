import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from datasets import Dataset, load_metric
import japanize_matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import time

class EmotionAnalysis:
    def __init__(self, use_japanese=False, use_gpu=False, use_plot=False) -> None:
        if use_japanese:
            self.emotion_names = ['喜び', '悲しみ', '期待', '驚き', '怒り', '恐れ', '嫌悪', '信頼']  # 日本語版
        else:
            self.emotion_names = ['Joy', 'Sadness', 'Anticipation', 'Surprise', 'Anger', 'Fear', 'Disgust', 'Trust']
        self.num_labels = len(self.emotion_names)
        if use_plot:
            sns.set(font='IPAGothic')
        tokenizer_checkpoint = 'cl-tohoku/bert-base-japanese-whole-word-masking'
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_checkpoint)
        model_checkpoint = 'emotion-analysis-model'
        self.model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=self.num_labels)
        if use_gpu:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")
        self.model.to(self.device)
        self.model.eval()

    def np_softmax(self, x):
        f_x = np.exp(x) / np.sum(np.exp(x))
        return f_x

    def analyze_emotion(self, text, show_fig=False, ret_prob=False):
        # 入力データ変換 + 推論
        tokens = self.tokenizer(text, truncation=True, return_tensors="pt").to(self.device)

        # モデルに入力し、GPU上で推論を実行する
        with torch.no_grad():
            preds = self.model(**tokens)
        prob = self.np_softmax(preds.logits.cpu().detach().numpy()[0])
        out_dict = {n: p for n, p in zip(self.emotion_names, prob)}

        # 棒グラフを描画
        if show_fig:
            plt.figure(figsize=(8, 3))
            df = pd.DataFrame(out_dict.items(), columns=['name', 'prob'])
            sns.barplot(x='name', y='prob', data=df)
            japanize_matplotlib.japanize()
            plt.title('入力文 : ' + text, fontsize=15)

        if ret_prob:
            max_label, max_value = max(out_dict.items(), key=lambda x: x[1])
            max_value = np.float64(max_value)
            return max_label, max_value
        
def main():
    # 動作確認
    model = EmotionAnalysis(use_japanese=False, use_gpu=True)
    start = time.time()
    result = model.analyze_emotion('今日から長期休暇だぁーーー！！！', show_fig=False, ret_prob=True)
    elapsed_time = time.time() - start
    print("elapsed_time:{:.3f}".format(elapsed_time))
    print(result)
    
if __name__ == "__main__":
    main() 