import torch
class ValueBuffer(object):
    """Buffer for outputting nonparametric q value"""
    def __init__(self, capacity, k,  key_width, n_actions, device, LRU=False):
        self.capacity = capacity
        self.neighbors = k
        self.key_width = key_width
        self.num_actions = n_actions
        self.device = device
        self.LRU = LRU

        if self.LRU:
            self.lru_timestamps = torch.empty(self.capacity, device=self.device, dtype=torch.int32)
            self.current_timestamp = 0
            self.i = 0
            self.embeddings = torch.empty((self.capacity, key_width), device=self.device, dtype=torch.float32)
            self.values = torch.empty((self.capacity, n_actions), device=self.device, dtype=torch.float32)
        else:
            self.embeddings = torch.empty((0, key_width), device=self.device, dtype=torch.float32)
            self.values = torch.empty((0, n_actions), device=self.device, dtype=torch.float32)

    def store(self, embeddings, values):
        if self.LRU:
            self._lru_store(embeddings, values)
        else:
            self._normal_store(embeddings, values)

    def _normal_store(self, embeddings, values):
        """Save Q value corresponding to emmbedding
        """
        self.embeddings = torch.cat((self.embeddings, embeddings), dim=0)[-self.capacity:]
        self.values = torch.cat((self.values, values), dim=0)[-self.capacity:]

    def _lru_store(self, embeddings, values):
        for embedding, value in zip(embeddings, values):
            if self.i < self.capacity:
                self.embeddings[self.i] = embedding
                self.values[self.i] = value
                self.lru_timestamps[self.i] = self.current_timestamp
                self.i += 1
                self.current_timestamp += 1
            else:
                index = torch.argmin(self.lru_timestamps)
                self.embeddings[index] = embedding
                self.values[index] = value
                self.lru_timestamps[index] = self.current_timestamp
                self.current_timestamp += 1
        return

    def get_q(self, embedding):
        indices = self.index_search(embedding)
        if self.LRU:
            for index in indices:
                self.lru_timestamps[index] = self.current_timestamp
                self.current_timestamp += 1
        values = torch.tensor(self.values[indices], device=self.device)
        return torch.mean(values, dim=0)

    def index_search(self, embedding):
        distances = torch.sum((torch.tensor(embedding, device=self.device) - self.embeddings)**2, dim=1)
        return torch.argsort(distances)[:self.neighbors]