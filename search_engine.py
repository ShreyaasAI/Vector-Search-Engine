import math
import numpy as np
from sentence_transformers import SentenceTransformer


class VectorCompare:
    

    def __init__(self, use_embeddings=True, model_name="all-MiniLM-L6-v2"):
        self.use_embeddings = use_embeddings
        if use_embeddings:
            # Downloads once, then caches locally. ~80MB, fast, runs on CPU.
            self.model = SentenceTransformer(model_name)

    # ----  semantic vectorization ----
    def encode(self, text):
        if self.use_embeddings:
            return self.model.encode(text, normalize_embeddings=True)
        return self.concordance(text)

    # ---- legacy: bag-of-words (kept for fallback / comparison) ----
    def concordance(self, document):
        if type(document) != str:
            raise ValueError('Supplied Argument should be of type string')
        con = {}
        for word in document.split(' '):
            con[word] = con.get(word, 0) + 1
        return con
    # ---standard vector magnitude, word standard deviation----
    def magnitude(self, concordance):
        if type(concordance) != dict:
            raise ValueError('Supplied Argument should be of type dict')
        total = sum(count ** 2 for count in concordance.values())
        return math.sqrt(total)

    # ---- cosine similarity, works for dict OR dense vector ----
    def relation(self, vec1, vec2):
        if isinstance(vec1, dict) and isinstance(vec2, dict):
            topvalue = sum(count * vec2.get(word, 0) for word, count in vec1.items())
            denom = self.magnitude(vec1) * self.magnitude(vec2)
            return topvalue / denom if denom else 0
        else:
            a, b = np.array(vec1), np.array(vec2)
            denom = np.linalg.norm(a) * np.linalg.norm(b)
            return float(np.dot(a, b) / denom) if denom else 0


v = VectorCompare(use_embeddings=True)

documents = {
    0: "The old lighthouse keeper watched storms roll across the harbor every winter, tracking wind speed, tide patterns, and the distant flicker of fishing boats returning home before dark.",
    1: "Investors debated whether rising interest rates, inflation, and global supply chain disruptions would trigger a recession or simply slow economic growth throughout the following fiscal year.",
    2: "Deep in the rainforest, biologists documented rare orchids, venomous frogs, and elusive jaguars while studying how deforestation threatens fragile ecosystems and biodiversity worldwide.",
    3: "After years of practice, the violinist finally mastered the concerto, blending precise technique with raw emotion to move the audience to tears during her farewell performance.",
    4: "Engineers redesigned the bridge using lightweight steel alloys and advanced computer simulations to withstand earthquakes, heavy traffic loads, and decades of harsh coastal weather.",
    5: "The chef combined saffron, garlic, and fresh basil into a rich tomato sauce, layering it with pasta, cheese, and herbs to create a comforting family dinner.",
    6: "Historians argued that the revolution stemmed from widespread poverty, political corruption, and a growing hunger for freedom among peasants, merchants, and disillusioned soldiers alike.",
    7: "During the marathon, runners battled exhaustion, dehydration, and blistering heat, pushing through pain with determination while spectators cheered and offered water along the route.",
    8: "The astronomer calibrated her telescope carefully, hoping to capture images of distant galaxies, pulsars, and nebulae before dawn light washed out the night sky.",
    9: "Parents juggled work deadlines, school pickups, grocery shopping, and household chores, finding brief moments of peace only after the children finally fell asleep.",
}


index = {i: v.encode(doc) for i, doc in documents.items()}

searchterm = input('Enter Search Term: ')
query_vec = v.encode(searchterm)

matches = []
for i, vec in index.items():
    score = v.relation(query_vec, vec)
    if score != 0:
        matches.append((score, documents[i][:100]))

matches.sort(reverse=True)

for score, snippet in matches:
    print(f"{score:.4f}  {snippet}")  