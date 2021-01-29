import numpy as np
from measurement_data import MeasurementData

class MeasurementFitter:
    def __init__(self, m: MeasurementData, std=4.):
        self.meas_data = m
        self.channel_data, self.ts = m.get_smoothened_relative_data(std)
        self.n_samples = self.ts.shape[0]

        # self.ts = (m.timestamps - m.timestamps[0]) / np.timedelta64(1, 's')

    def fit_drift(self, delta_t: int):
        """
        fit straights to all data ranges of length in the channel data
        :param delta_t: time span in seconds used for each straight
        :return: m: array of slopes,
                 b: array of offsets,
                 noise: std deviation of delta between predicted values by fitted straight and the real values
        """
        m = []
        b = []
        noise = []
        for start in range(0, self.n_samples-int(delta_t/2)):
            # prepare data for data range starting at index start
            x = []
            y = []
            i = start
            while i < self.n_samples and self.ts[i] < self.ts[start] + delta_t:
                x.append(self.ts[i])
                y.append(self.channel_data[i].reshape(1, -1))

                i += 1

            y = np.concatenate(y, axis=0)

            # fit straight
            coeffs = np.polyfit(x, y, deg=1)
            m.append(coeffs[0].reshape((1, -1)))
            b.append(coeffs[1].reshape((1, -1)))

            x_pred = m[-1] * np.array(x).reshape((-1, 1)) + b[-1]
            delta_y = x_pred - y
            noise.append(np.std(delta_y, axis=0).reshape((1, -1)))

        m = np.concatenate(m, axis=0)
        b = np.concatenate(b, axis=0)
        noise = np.concatenate(noise, axis=0)
        return m, b, noise

    def get_slope_diffs(self, delta_t):
        """
        Calculates slope differences for each sample with at least delta_t / 3 previous and subsequent samples.
        The following steps are executed for each sample:
        1) Check if sample has at least delta_t / 3  previous and subsequent samples within delta_t time difference.
           If so, continue with 2), check next sample otherwise
        2) Fit straights to the previous and subsequent samples within delta_t respectively
        3) Calculate the slope diffs between the previous and subsequent fitted straights
        :param delta_t: time span in seconds used for each straight
        :return:
        """
        m_diff = []
        b_diff = []
        t = []

        for sample_index in range(self.n_samples):
            # 1) collect and check previous and subsequent samples:
            # collect previous samples
            prev_x = []
            prev_y = []
            i = sample_index-1
            while i >= 0 and self.ts[i] > self.ts[sample_index] - delta_t:
                prev_x.append(self.ts[i])
                prev_y.append(self.channel_data[i].reshape(1, -1))

                i -= 1
            if len(prev_y) < delta_t / 3:   # next sample if not enough previous samples
                continue
            prev_y = np.concatenate(prev_y, axis=0)

            # collect subsequent samples
            subseq_x = []
            subseq_y = []
            i = sample_index+1
            while i < self.n_samples and self.ts[i] < self.ts[sample_index] + delta_t:
                subseq_x.append(self.ts[i])
                subseq_y.append(self.channel_data[i].reshape(1, -1))

                i += 1
            if len(subseq_y) < delta_t / 3:  # next sample if not enough subsequent samples
                continue
            subseq_y = np.concatenate(subseq_y, axis=0)

            # 2) fit straights:
            # previous samples
            coeffs = np.polyfit(prev_x, prev_y, deg=1)
            m_prev = coeffs[0].reshape((1, -1))
            b_prev = coeffs[1].reshape((1, -1))

            # subsequent samples
            coeffs = np.polyfit(subseq_x, subseq_y, deg=1)
            m_subseq = coeffs[0].reshape((1, -1))
            b_subseq = coeffs[1].reshape((1, -1))

            m_diff.append(m_subseq - m_prev)
            b_diff.append(b_subseq - b_prev)
            t.append(self.ts[sample_index])

        m_diff = np.concatenate(m_diff, 0)
        b_diff = np.concatenate(b_diff, 0)

        return m_diff, b_diff, t





if __name__ == "__main__":
    # meas_data = MeasurementData("/home/pingu/eNose-ml-engine/data/eNose-base-dataset/train/5_Ammoniak_200206.csv")
    meas_data = MeasurementData(
        "/home/pingu/Cloud/Studium/Studienarbeit/data/19CHButterfly_litho_1mmp/50µm/24 ml NH3/"
        "chip 1_4_33_33ml Zelle.csv")
    f = MeasurementFitter(meas_data)
    m, b, noise = f.fit_drift(20)
    m_diff, b_diff, t = f.get_slope_diffs(20)
