import json
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import messages_to_dict
from langchain.prompts.chat import (
    ChatPromptTemplate,
    MessagesPlaceholder, 
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

class LangChainChat:
    def __init__(self):
        # ChatOpenAIクラスのインスタンスを作成、temperatureは0.7を指定
        load_dotenv()
        self.chat = ChatOpenAI(temperature=0.7)
        # 会話の履歴を保持するためのオブジェクト
        self.memory = ConversationBufferMemory(return_messages=True)
        # テンプレートを作成
        self.prompt = self.make_template()
        self.conversation = ConversationChain(llm=self.chat, memory=self.memory, prompt=self.prompt)

    def make_template(self):
        # テンプレートを定義
        template = """  
        以下は、人間とAIのフレンドリーな会話です。
        AIはその文脈から具体的な内容をたくさん教えてくれます。
        AIは質問の答えを知らない場合、正直に「知らない」と答えます。
        """

        # テンプレートを用いてプロンプトを作成
        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(template),
            MessagesPlaceholder(variable_name="history"),
            HumanMessagePromptTemplate.from_template("{input}")
        ])

        return prompt

    # AIとの会話を管理
    def conversation(self, command):
       # ユーザからの入力を受け取り、変数commandに代入
       response = self.conversation.predict(input=command)
       # memoryオブジェクトから会話の履歴を取得して、変数historyに代入
       history = self.memory.chat_memory
       # 会話履歴をJSON形式の文字列に変換、整形して変数messagesに代入
       messages = json.dumps(messages_to_dict(history.messages), indent=2, ensure_ascii=False)
       # 会話履歴を表示
       return {"response": response, "message": messages}