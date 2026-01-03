class GenerationSession:
    
    def __init__(self, ldm, batch_size: int):
        self.ldm = ldm
        self.batch_size = batch_size
        self.step = 0
        self.z = self.ldm.sample_api_init(batch_size)

    def next(self):
        if self.step >= self.ldm.time_steps:
            return None, None

        self.z, x = self.ldm.sample_api_step(self.z, self.step)
        self.step += 1
        return self.step, x
