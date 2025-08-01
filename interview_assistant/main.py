import argparse
import yaml
from pathlib import Path
import pyttsx3
import speech_recognition as sr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from transformers import pipeline


class KnowledgeBase:
    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self.docs = []
        self.sources = []
        for folder in self.base_dir.iterdir():
            if folder.is_dir():
                for file in folder.glob("*.txt"):
                    self.docs.append(file.read_text())
                    self.sources.append(str(file))
        self.vectorizer = TfidfVectorizer().fit(self.docs)
        self.doc_matrix = self.vectorizer.transform(self.docs)

    def search(self, query: str, top_k: int = 3):
        if not self.docs:
            return []
        q_vec = self.vectorizer.transform([query])
        scores = linear_kernel(q_vec, self.doc_matrix).flatten()
        ranked = scores.argsort()[::-1][:top_k]
        return [self.docs[i] for i in ranked if scores[i] > 0]


class InterviewAssistant:
    def __init__(self, kb_dir: str, config_path: str, model_name: str = "gpt2"):
        self.kb = KnowledgeBase(kb_dir)
        self.style = yaml.safe_load(Path(config_path).read_text())
        self.generator = pipeline("text-generation", model=model_name)
        self.tts = pyttsx3.init()

    def build_prompt(self, query: str, context: list[str]):
        style = self.style.get("answer_style", {})
        tone = style.get("tone", "")
        length = style.get("length", "")
        focus = style.get("focus", "")
        persona = style.get("persona", "")
        fmt = style.get("format", "")
        context_text = "\n".join(context)
        prompt = (
            f"You are {persona}. Use a {tone} tone. Keep the answer {length}. "
            f"Focus on {focus}. Format: {fmt}.\n\n"
            f"Context:\n{context_text}\n\nQuestion: {query}\nAnswer:"
        )
        return prompt

    def generate_answer(self, query: str):
        context = self.kb.search(query)
        prompt = self.build_prompt(query, context)
        result = self.generator(prompt, max_length=256, do_sample=False)[0]["generated_text"]
        return result[len(prompt) :].strip()

    def speak(self, text: str):
        self.tts.say(text)
        self.tts.runAndWait()


def transcribe_audio(path: str) -> str:
    recognizer = sr.Recognizer()
    with sr.AudioFile(path) as source:
        audio = recognizer.record(source)
    return recognizer.recognize_google(audio)


def main():
    parser = argparse.ArgumentParser(description="Interview Assistant")
    parser.add_argument("--audio", type=str, help="Path to question wav file")
    args = parser.parse_args()
    kb_dir = Path(__file__).parent / "knowledge_base"
    cfg = Path(__file__).parent / "config.yaml"
    assistant = InterviewAssistant(kb_dir, cfg)

    if args.audio:
        question = transcribe_audio(args.audio)
    else:
        question = input("Question: ")
    answer = assistant.generate_answer(question)
    print(answer)
    assistant.speak(answer)


if __name__ == "__main__":
    main()
