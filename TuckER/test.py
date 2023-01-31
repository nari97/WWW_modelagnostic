from tucker import TuckER
from load_data import Data

if __name__ == "__main__":
    d = Data(data_dir="data/WN18RR/")
    e_dim = 200
    r_dim = 30

    model = TuckER(d, e_dim, r_dim, input_dropout=0.1, hidden_dropout1=0.2, hidden_dropout2=0.2)
    print (d.entities)
    print (d.relations)