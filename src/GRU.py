
import argparse
import sys

from Features import DataProcess
from RNNmodel import GruRnn, LSTMRnn, RecurrentModel

if __name__ == '__main__':
  parser = argparse.ArgumentParser(prog='LSTM.py', description='LSTM model')
  parser.add_argument('-i','--path', required=True, help='input path for data')
  parser.add_argument('-d','--use_date', required=False, default=False, action='store_true', help='use date in feature')
  parser.add_argument('-e','--use_encoded_date', required=False, default=False, action='store_true', help='use embedded date in feature')
  parser.add_argument('-p','--predict_x_hours', required=False, default=24, type=int, help='value to predict for next x hours')
  oj_args = parser.parse_args(sys.argv[1:])

  print("INFO: model will predict for {} hours' temperature".format(oj_args.predict_x_hours))
  in_predict_value = oj_args.predict_x_hours * 6  

  oj_kmodel = GruRnn(0.80, 0.10, 0.10, 2016, in_predict_value, 6, 128)
  oj_kmodel.read_data(oj_args.path, use_date=oj_args.use_date, use_embedded_date=oj_args.use_encoded_date)
  oj_kmodel.identify_outlier(sigma = 5.0)
  oj_kmodel.fit_standard_scalar()

  if (not oj_args.use_date and not oj_args.use_encoded_date) or \
     (oj_args.use_date and not oj_args.use_encoded_date):
    oj_kmodel.input_model()
    oj_kmodel.keras_gru()
    oj_kmodel.fit_model_generator()
  elif not oj_args.use_date and oj_args.use_encoded_date:
    oj_kmodel.merge_model()
    oj_kmodel.keras_gru_merge_model()
    oj_kmodel.fit_model_generator_merge_model()
  else:
    print("INFO: invalid selection!!!")
    sys.exit(0)
