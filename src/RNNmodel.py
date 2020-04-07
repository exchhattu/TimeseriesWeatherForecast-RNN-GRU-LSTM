import numpy as np
from keras import optimizers, regularizers
from keras.layers import GRU, LSTM, Dense, Input, concatenate
from keras.layers.core import Activation, Dense, Dropout, Reshape
from keras.layers.embeddings import Embedding
from keras.models import Model, Sequential
from keras.regularizers import L1L2

from Features import DataProcess


class RecurrentModel(DataProcess):
    def __init__(
        self,
        fo_train_ratio,
        fo_valid_ratio,
        fo_test_ratio,
        in_back_offset=2016,
        in_forward_offset=72,
        in_step=6,
        in_batch_size=128,
    ):
        """ setup hyperparameters of RNN and parameters to split data 
        
        Args:
            fo_train_ratio: ratio to put data in training dataset 
            fo_valid_ratio: ratio to put data in validation dataset 
            fo_test_ratio:  ratio to put data in testing dataset 
            in_back_offset: # of records prior to current data to include for a data. Defaults to 2016.
            in_forward_offset: for which day/hour from now, temperature would be predicted. Defaults to 72.
            in_step: step size was set to reduce a redundancy. Defaults to 6.
            in_batch_size: define batch size. Defaults to 128.
        """
        super().__init__(fo_train_ratio, fo_valid_ratio, fo_test_ratio)
        self._in_back_offset = in_back_offset
        self._in_forward_offset = in_forward_offset
        self._in_step = in_step
        self._in_batch_size = in_batch_size

    def data_generator(
        self,
        in_min_idx=0,
        in_max_idx=120000,
        in_batch_size=128,
        random=True,
        valid="Test",
    ):
        """ Batch data generation. Since it is a time series data, order of 
            record is important. Order is maintained since future information 
            cannot be used to predict past activity. 
        
        Args:
            in_min_idx: starting index for batch. Defaults to 0.
            in_max_idx: ending index. Defaults to 120000.
            in_batch_size: batch size. Defaults to 128.
            random: random selection of data. Defaults to True.
            valid: type of data either validation/test. Defaults to "Test".
        """
        if in_max_idx is None:
            in_max_idx = self._ar_values.shape[0] - self._in_back_offset - 1
        in_idx = in_min_idx + self._in_back_offset
        in_b_dim = self._in_back_offset // self._in_step

        while True:
            if in_idx > in_max_idx - (self._in_batch_size + self._in_forward_offset):
                in_idx = in_min_idx + self._in_back_offset

            ar_rows = []
            if random:
                ar_rows = np.random.randint(
                    in_idx,
                    in_max_idx - self._in_forward_offset,
                    size=self._in_batch_size,
                )
            else:
                ar_rows = np.arange(
                    in_idx, min(in_max_idx, in_idx + self._in_batch_size)
                )

            in_num_rows = len(ar_rows)
            ar_X_f = np.zeros(
                (in_num_rows, in_b_dim, self._ts_num_idxes.shape[0]), dtype=float
            )
            ar_Y = np.zeros((in_num_rows, 1))

            in_j = 0
            while in_j < in_num_rows:
                indices = np.arange(
                    ar_rows[in_j], ar_rows[in_j] + self._in_back_offset, self._in_step
                )
                ar_X_f[in_j] = self._ar_values[indices[:, None], self._ts_num_idxes]
                ar_Y[in_j] = self._ar_values[
                    ar_rows[in_j] + self._in_back_offset + self._in_forward_offset,
                    self._in_target_idx,
                ]
                in_j += 1
            in_idx += in_num_rows

            """ for debugging
      print("DEBUG: ar_X_f  ", valid, ar_X_f[0:2, 2:4,:])
      print("DEBUG: ar_X min ", ar_X_f.min())
      print("DEBUG: ar_X max ", ar_X_f.max())
      print("DEBUG: ar_Y min ", ar_Y.min())
      print("DEBUG: ar_Y max ", ar_Y.max())
      """
            yield ar_X_f, ar_Y

    def fit_standard_scalar(self):
        """ Normalized the train data and parameter derived from it 
            will be used same paramters for test and validation set
        """
        in_num_sample = int(self._fo_train_ratio * self._ar_values.shape[0])
        ar_mean = (
            self._ar_values[:in_num_sample, self._ts_num_idxes]
            .astype(np.float64)
            .mean(axis=0)
        )
        self._ar_values[:, self._ts_num_idxes] -= ar_mean
        ar_std = (
            self._ar_values[:in_num_sample, self._ts_num_idxes]
            .astype(np.float64)
            .std(axis=0)
        )
        self._ar_values[:, self._ts_num_idxes] /= ar_std

        print(self._in_target_idx)

    def merge_data_generator(
        self,
        in_min_idx=0,
        in_max_idx=120000,
        in_batch_size=128,
        random=False,
        valid="test",
    ):
        """ Merge columns with numeric and string data types but 
            make sure columns with string data types should be converted 
            to create final tensor 
        
        Args:
            in_min_idx: start index for batch. Defaults to 0.
            in_max_idx: end index. Defaults to 120000.
            in_batch_size: size of batch. Defaults to 128.
            random: random selection. Defaults to False.
            valid: type of data used either test and valid. Defaults to "test".
        """
        self._ar_test = []
        if in_max_idx is None:
            in_max_idx = (
                self._ar_values.shape[0] - self._in_back_offset - 1
            )  # self._in_forward_offset - 1
        in_idx = in_min_idx + self._in_back_offset
        in_b_dim = self._in_back_offset // self._in_step

        while True:
            if in_idx > in_max_idx - (self._in_batch_size + self._in_forward_offset):
                in_idx = np.random.randint(
                    self._in_back_offset,
                    in_max_idx - (self._in_batch_size + self._in_forward_offset),
                )

            ar_rows = []
            if random:
                ar_rows = np.random.randint(
                    in_idx,
                    in_max_idx - self._in_forward_offset,
                    size=self._in_batch_size,
                )
            else:
                ar_rows = np.arange(
                    in_idx, min(in_max_idx, in_idx + self._in_batch_size)
                )

            in_num_rows = len(ar_rows)

            ar_X_f = np.zeros((in_num_rows, in_b_dim, self._ts_num_idxes.shape[0]))
            ar_X1_s = np.zeros((in_num_rows, in_b_dim, 1))
            ar_X2_s = np.zeros((in_num_rows, in_b_dim, 1))
            ar_X3_s = np.zeros((in_num_rows, in_b_dim, 1))
            ar_X4_s = np.zeros((in_num_rows, in_b_dim, 1))
            ar_Y = np.zeros((in_num_rows, 1))

            in_j = 0
            while in_j < in_num_rows:
                indices = np.arange(
                    ar_rows[in_j], ar_rows[in_j] + self._in_back_offset, self._in_step
                )
                ar_X_f[in_j] = self._ar_values[indices[:, None], self._ts_num_idxes]
                ar_X1_s[in_j] = self._ar_values[indices[:, None], self._ts_str_idxes[0]]
                ar_X2_s[in_j] = self._ar_values[indices[:, None], self._ts_str_idxes[1]]
                ar_X3_s[in_j] = self._ar_values[indices[:, None], self._ts_str_idxes[2]]
                ar_X4_s[in_j] = self._ar_values[indices[:, None], self._ts_str_idxes[3]]

                ar_Y[in_j] = self._ar_values[
                    ar_rows[in_j] + self._in_back_offset + self._in_forward_offset,
                    self._in_target_idx,
                ]
                if valid == "test":
                    self._ar_test += ar_Y[in_j]
                in_j += 1
            in_idx += in_num_rows
            yield [ar_X_f, ar_X1_s, ar_X2_s, ar_X3_s, ar_X4_s], ar_Y  # .ravel()

    def input_model(self):
        """ Create input models using features that are numeric 
        """
        in_num_features = len(self._ts_num_idxes)
        in_b_dim = self._in_back_offset // self._in_step
        self._model = Input(shape=(in_b_dim, in_num_features,), name="n_input")

    def merge_model(self):
        """ merge two models to create input model. One model is from numeric features
            another model is from categorical or other data types (non-numeric)  
        """

        in_num_features = len(self._ts_num_idxes)
        in_b_dim = self._in_back_offset // self._in_step
        self._oj_num_input = Input(shape=(in_b_dim, in_num_features,), name="n_input")

        """categorical columns"""
        in_str_features = self._ts_str_idxes
        self._st_input_dim = [2020, 13, 32, 1500]  # time, year, month, day
        self._st_output_dim = [4, 4, 5, 20]  # time, year, month, day
        self._ts_st_inputs = []
        ts_st_models = []

        for i in range(len(self._st_input_dim)):
            st_st_cname = "s_input_%s" % str(i)
            oj_st_input = Input(shape=(in_b_dim, 1), name=st_st_cname)
            self._ts_st_inputs.append(oj_st_input)
            oj_st_model = Embedding(self._st_input_dim[i], self._st_output_dim[i])(
                oj_st_input
            )
            ts_st_models.append(oj_st_model)

        oj_st_merged_em_model = concatenate(ts_st_models)
        self._st_merged_em_model = Reshape((in_b_dim, -1))(oj_st_merged_em_model)

        """merge numerical and embedding model"""
        self._model = concatenate([self._oj_num_input, self._st_merged_em_model])

    def dump_history(self, ma_history_log, st_out_fname):
        """ Save metrices for post processing. This data 
            is useful for plotting graphs to see epoch vs loss function 
        
        Args:
            ma_history_log: path to output the log files 
            st_out_fname: output file name 
        """

        with open(st_out_fname, "w") as oj_written:
            # format in output file - three columns
            # epoch,key1,key2
            ts_val_losses = ma_history_log["val_loss"]
            ts_losses = ma_history_log["loss"]
            oj_written.write("%s,%s,%s\n" % ("epoch", "loss", "val_loss"))

            in_epoch = 1
            for fo_loss, fo_val_loss in zip(ts_losses, ts_val_losses):
                oj_written.write("%d,%.5f,%.5f\n" % (in_epoch, fo_loss, fo_val_loss))
                in_epoch += 1

    def fit_model_generator(self, out_ftag="GRU"):
        """ this generates a model using features with numeric values 
        
        Args:
            out_ftag: log file name. This is used to represent the log of LSTM or GRU. 
                      Defaults to "GRU".
        """

        in_num_sample = self._ar_values.shape[0]
        in_lbound = 0
        in_ubound = int(self._fo_train_ratio * in_num_sample)
        ge_train = self.data_generator(0, in_ubound, self._in_batch_size, True, "tra")
        print(
            "INFO: training {}: {} from {} to {}".format(
                self._fo_train_ratio, in_ubound - in_lbound, in_lbound, in_ubound
            )
        )

        in_lbound, in_ubound = (
            in_ubound + 1,
            int(in_ubound + self._fo_valid_ratio * in_num_sample),
        )
        ge_valid = self.data_generator(
            in_lbound, in_ubound, self._in_batch_size, False, "val"
        )
        in_valid_step = in_ubound - in_lbound - self._in_back_offset
        print(
            "INFO: validation {}: {} from {} to {}".format(
                self._fo_valid_ratio, in_ubound - in_lbound, in_lbound, in_ubound
            )
        )

        in_lbound, in_ubound = (
            in_ubound + 1,
            int(in_ubound + self._fo_test_ratio * in_num_sample),
        )
        ge_test = self.data_generator(
            in_lbound, None, self._in_batch_size, False, "test"
        )
        in_test_step = in_ubound - in_lbound - self._in_back_offset
        print(
            "INFO: testing {}: {} from {} to {}".format(
                self._fo_test_ratio, in_ubound - in_lbound, in_lbound, in_ubound
            )
        )

        ma_seq_model_log = self._fmodel.fit_generator(
            ge_train,
            steps_per_epoch=200,
            epochs=20,
            validation_data=ge_valid,
            validation_steps=in_valid_step // self._in_batch_size,
            use_multiprocessing=False,
            shuffle=False,
        )
        self.dump_history(ma_seq_model_log.history, "%s_hist.log" % out_ftag)
        print("Performance on independent TEST:")
        print("--------------------------------")
        oj_predict = self._fmodel.evaluate_generator(
            ge_test,
            steps=in_test_step // self._in_batch_size,
            use_multiprocessing=False,
        )
        print(f"{self._fmodel.metrics_names[0]:20s}:{oj_predict[0]:0.2f}")
        print(f"{self._fmodel.metrics_names[1]:20s}:{oj_predict[1]:0.2f}")

    def fit_model_generator_merge_model(self, out_ftag="GRU"):
        """ generate a model that uses all features (numeric and categorical)
        
        Args:
            out_ftag: tag for output log file based on architecture. Defaults to "GRU".
        """

        in_num_sample = self._ar_values.shape[0]
        in_lbound = 0
        in_ubound = int(self._fo_train_ratio * in_num_sample)
        ge_train = self.merge_data_generator(0, in_ubound, self._in_batch_size, "tra")
        print(
            "INFO: training {}: {} from {} to {}".format(
                self._fo_train_ratio, in_ubound - in_lbound, in_lbound, in_ubound
            )
        )

        in_lbound, in_ubound = (
            in_ubound + 1,
            int(in_ubound + self._fo_valid_ratio * in_num_sample),
        )
        ge_valid = self.merge_data_generator(
            in_lbound, in_ubound, self._in_batch_size, "val"
        )
        in_valid_step = in_ubound - in_lbound - self._in_back_offset
        print(
            "INFO: validation {}: {} from {} to {}".format(
                self._fo_valid_ratio, in_ubound - in_lbound, in_lbound, in_ubound
            )
        )

        in_lbound, in_ubound = (
            in_ubound + 1,
            int(in_ubound + self._fo_test_ratio * in_num_sample),
        )
        ge_test = self.merge_data_generator(
            in_lbound, None, self._in_batch_size, "test"
        )
        in_test_step = in_ubound - in_lbound - self._in_back_offset
        print(
            "INFO: testing {}: {} from {} to {}".format(
                self._fo_test_ratio, in_ubound - in_lbound, in_lbound, in_ubound
            )
        )

        ma_seq_model_log = self._fmodel.fit_generator(
            ge_train,
            steps_per_epoch=200,
            epochs=20,
            validation_data=ge_valid,
            validation_steps=in_valid_step // self._in_batch_size,
            use_multiprocessing=False,
            shuffle=False,
        )
        self.dump_history(ma_seq_model_log.history, "%s_merge_hist.log" % out_ftag)
        print("Performance on independent TEST:")
        print("--------------------------------")
        oj_predict = self._fmodel.evaluate_generator(
            ge_test,
            steps=in_test_step // self._in_batch_size,
            use_multiprocessing=False,
        )
        print(f"{self._fmodel.metrics_names[0]:20s}:{oj_predict[0]:0.2f}")
        print(f"{self._fmodel.metrics_names[1]:20s}:{oj_predict[1]:0.2f}")


class GruRnn(RecurrentModel):
    """ Child class inherited from RecurrentModel
    """

    def __init__(
        self,
        fo_train_ratio,
        fo_valid_ratio,
        fo_test_ratio,
        in_back_offset=2016,
        in_forward_offset=72,
        in_step=6,
        in_batch_size=128,
    ):
        """ Specific to GRU architecture 
        
        Args:
            fo_train_ratio: ratio to put data in training dataset 
            fo_valid_ratio: ratio to put data in validation dataset 
            fo_test_ratio:  ratio to put data in testing dataset 
            in_back_offset: # of records prior to current data to include for a data. Defaults to 2016.
            in_forward_offset: for which day/hour from now, temperature would be predicted. Defaults to 72.
            in_step: step size was set to reduce a redundancy. Defaults to 6.
            in_batch_size: define batch size. Defaults to 128.
        """

        super().__init__(
            fo_train_ratio,
            fo_valid_ratio,
            fo_test_ratio,
            in_back_offset,
            in_forward_offset,
            in_step,
            in_batch_size,
        )

    def keras_gru(self):
        """ Prepare a GRU model with sigmoid activation function, 
            other activation should also be tried problem. 
            Noneless output layer is linear activation. 
        """

        oj_gru_model = GRU(
            18,
            activation="sigmoid",
            return_sequences=True,
            dropout=0.03,
            recurrent_dropout=0.03,
        )(self._model)
        oj_gru_model_01 = GRU(9, activation="sigmoid")(oj_gru_model)
        oj_dense_model = Dense(1)(oj_gru_model_01)
        self._fmodel = Model(self._model, oj_dense_model)
        self._fmodel.summary()
        self._fmodel.compile(
            optimizer=optimizers.RMSprop(), loss="mean_squared_error", metrics=["mse"]
        )

    def keras_gru_merge_model(self):
        """GRU model for two different types of data 
        """

        oj_gru_model = GRU(
            18,
            activation="sigmoid",
            return_sequences=True,
            dropout=0.03,
            recurrent_dropout=0.03,
        )(self._model)
        oj_gru_model_01 = GRU(9, activation="sigmoid")(oj_gru_model)
        oj_dense_model = Dense(1)(oj_gru_model_01)
        self._fmodel = Model([self._oj_num_input] + self._ts_st_inputs, oj_dense_model)
        self._fmodel.summary()
        self._fmodel.compile(
            optimizer=optimizers.RMSprop(), loss="mean_squared_error", metrics=["mse"]
        )


class LSTMRnn(RecurrentModel):
    def __init__(
        self,
        fo_train_ratio,
        fo_valid_ratio,
        fo_test_ratio,
        in_back_offset=2016,
        in_forward_offset=72,
        in_step=6,
        in_batch_size=128,
    ):
        """ LSTM architecture inherited from parent class RecurrentModel
            
        
        Args:
            fo_train_ratio: ratio to put data in training dataset 
            fo_valid_ratio: ratio to put data in validation dataset 
            fo_test_ratio:  ratio to put data in testing dataset 
            in_back_offset: # of records prior to current data to include for a data. Defaults to 2016.
            in_forward_offset: for which day/hour from now, temperature would be predicted. Defaults to 72.
            in_step: step size was set to reduce a redundancy. Defaults to 6.
            in_batch_size: define batch size. Defaults to 128.
        """
        super().__init__(
            fo_train_ratio,
            fo_valid_ratio,
            fo_test_ratio,
            in_back_offset,
            in_forward_offset,
            in_step,
            in_batch_size,
        )

    def keras_lstm(self):
        """ LSTM architecture for numeric feature 
        """

        oj_lstm_model = LSTM(
            18,
            activation="sigmoid",
            return_sequences=True,
            dropout=0.03,
            recurrent_dropout=0.03,
        )(self._model)
        oj_lstm_model = LSTM(8, activation="sigmoid")(oj_lstm_model)
        oj_dense_model = Dense(1)(oj_lstm_model)
        self._fmodel = Model(self._model, oj_dense_model)
        self._fmodel.summary()
        self._fmodel.compile(
            optimizer=optimizers.RMSprop(), loss="mean_squared_error", metrics=["mse"]
        )

    def keras_lstm_merge_model(self):
        """ LSTM architecture 
        """

        oj_lstm_model_01 = LSTM(
            18,
            activation="sigmoid",
            return_sequences=True,
            dropout=0.03,
            recurrent_dropout=0.03,
        )(self._model)
        oj_lstm_model = LSTM(8, activation="sigmoid")(oj_lstm_model_01)
        oj_dense_model = Dense(1)(oj_lstm_model)
        self._fmodel = Model([self._oj_num_input] + self._ts_st_inputs, oj_dense_model)
        self._fmodel.summary()
        self._fmodel.compile(
            optimizer=optimizers.RMSprop(), loss="mean_squared_error", metrics=["mse"]
        )
