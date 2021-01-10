import collections
import heapq


class Element:
    def __init__(self, count, word):
        self.count = count
        self.word = word

    def __lt__(self, other):
        if self.count == other.count:
            return self.word > other.word
        return self.count < other.count

    def __eq__(self, other):
        return self.count == other.count and self.word == other.word


def topKFrequent(words, k):
    counts = collections.Counter(words)

    freqs = []
    heapq.heapify(freqs)
    for word, count in counts.items():
        heapq.heappush(freqs, (Element(count, word), word))
        if len(freqs) > k:
            heapq.heappop(freqs)

    return [i[1] for i in heapq.nlargest(k, freqs)]


words = ["i", "love", "you", "i", "love", "coding", "i", "like", "sports"]
k = 2
print(topKFrequent(words, k))
