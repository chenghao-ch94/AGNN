import torch
import numpy as np


class CategoriesSampler():

    def __init__(self, label, n_batch, n_cls, n_per, ep_per_batch=1):
        self.n_batch = n_batch
        self.n_cls = n_cls
        self.n_per = n_per
        self.ep_per_batch = ep_per_batch

        label = np.array(label)
        self.catlocs = []
        for c in range(max(label) + 1):
            self.catlocs.append(np.argwhere(label == c).reshape(-1))
        # for i in range(len(self.catlocs)):
        #     print(len(self.catlocs[i]))

    def __len__(self):
        return self.n_batch
    
    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch = []
            #batch_t = []
            for i_ep in range(self.ep_per_batch):
                episode = []
                #episode_t = []
                classes = np.random.choice(len(self.catlocs), self.n_cls,
                                           replace=False)
                for c in classes:
                    # print(len(self.catlocs[c]))
                    l = np.random.choice(self.catlocs[c], self.n_per,
                                         replace=False)
                    #l2 = self.catlocs[c] 
                    episode.append(torch.from_numpy(l))
                    #episode_t.append(torch.from_numpy(l2))
                episode = torch.stack(episode)
                #episode_t = torch.stack(episode_t)
                batch.append(episode)
                #batch_t.append(episode_t)
            batch = torch.stack(batch) # bs * n_cls * n_per
            #batch_t = torch.stack(batch_t)
            #yield batch.view(-1)
            yield batch.view(-1) #, batch_t.sview(-1)

