from train_data import TrainData
from fractal_gen import FractalGenTensorflowModel

if __name__ == '__main__':
    while True:
        t = TrainData.get_random()
        t.get_train_image()
        t.get_next_train_image()


# Sample class for generating a sequence of fractal images
class FractalGenModel():
    def __init__(self):        
        #input params
        ratio = 9.0 / 16.0
        self.start_x = -2.3
        self.end_x = self.start_x * -1.
        self.start_y = self.start_x * ratio
        self.end_y = self.end_x * ratio
        self.width = 1920 # image width
        self.bg_ratio = (4, 2.5, 1) # background color ratio
        self.ratio = (0.9, 0.9, 0.9)

        step = (self.end_x - self.start_x) / self.width
        Y, X = np.mgrid[self.start_y:self.end_y:step, self.start_x:self.end_x:step]
        self._Z = X + 1j * Y

        self._tfmodel = FractalGenTensorflowModel(self._Z.shape, self._Z.dtype)

    def generate_sequence(self, iters=600):
        for i in range(0, iters):
            theta = 2 * np.pi * i / iters
            # c = -(0.835 - 0.1 * np.cos(theta)) - (0.2321 + 0.1 * np.sin(theta)) * 1j
            # c = -0.8 * 1j
            c = -(0.835 - 0.05 * np.cos(theta)) - (0.2321 + 0.05 * np.sin(theta)) * 1j
            y = self._tfmodel.generate_image(self._Z, c, self.bg_ratio, self.ratio)
            y.save(f'.data/imgs/f-{i}.png', 'PNG')
            print(f'{datetime.now()} Fractal gen completed {i+1}/{iters}')
            yield y
