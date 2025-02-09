import ollama
import faiss
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class TextGenerator:
    answers = [
        "ラジオ体操は、ラジオを聴きながら体操を行います",
        "ラジオは、チャンネルを合わせて受信します",
        "テレビ体操は、テレビでラジオ体操を放送しています",
        "サラリーマン体操は、テレビでサラリーマンが体操します"
    ]

    answers_vector = []

    def __init__(self, client, model):
        self.client = client
        self.model = model
        # 回答を取得
        self.answers_vector = [self.vectorize_text(answer) for answer in self.answers]
        # ベクトルの定義 (任意の数のベクトル)
        self.answers_vector = [self.vectorize_text(answer) for answer in self.answers]
        # ベクトルをファイルにシリアライズする
        np.save('vectors.npy', self.answers_vector)

    def generate(self, retrieved_data):
        prompt = f""

    def vectorize_text(self, text):
        response = ollama.embeddings(
            model="mxbai-embed-large",
            prompt=text
        )
        return response["embedding"]

    def rag_sample(self, question):

        # 質問を取得
        question_vector = self.vectorize_text(question)

        # コサイン類似度が最も高い回答を取得
        max_similarity = 0
        most_similar_index = 0
        answers_vector = np.load('vectors.npy')
        for index, vector in enumerate(answers_vector):
            similarity = cosine_similarity([question_vector], [vector])[0][0]
            print(f"コサイン類似度: {similarity.round(4)}:{self.answers[index]}")
            # 取り出したコサイン類似度が最大のものを保存
            if similarity > max_similarity:
                max_similarity = similarity
                most_similar_index = index
        print(f"\n質問: {question}\n回答: {self.answers[most_similar_index]}\n")
        print(f"最も類似度が高い回答: {self.answers[most_similar_index]}¥n")

        prompt = f'''
次の質問について、情報を基に簡潔に回答して下さい。
# 質問
{question}
# 情報
{self.answers[most_similar_index]}
# 回答
'''
        print(prompt)
        stream = self.client.generate(
            model=self.model,
            prompt=prompt,
            stream=True
        )
        for chunk in stream:
            c = chunk['response']
            print(c, end='', flush=False)
        print("\n")

if __name__ == '__main__':
    # 質問文
    question1 = "サラリーマンは何体操をしますか？"
    question2 = "ラジオ体操は何を聴きながら体操しますか？"
    question3 = "ラジオ体操を放送しているメディアは？"
    generator = TextGenerator(client=ollama, model="vanilj/Phi-4:Q4_K_M")
    generator.rag_sample(question1)
    generator.rag_sample(question2)
    generator.rag_sample(question3)
