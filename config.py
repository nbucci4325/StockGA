"""
Central configuration for the entire project (pre-GA + GA).

Every hardcoded constant to be used for data fetching, feature engineering,
model training, and the GA is here, separated by their own dataclasses.
"""

from dataclasses import dataclass, field



@dataclass(frozen=True)
class IndicatorConfig:
    """
    IndicatorConfig describes what kind of stock/regime is being analyzed for the neural network. 
    Since the NN is initially trained across amny different stocks, these periods must be identical for all of them.
    """
    rsi_period: int = 14

    sma_short_period: int = 50
    sma_long_period: int = 200

    macd_fast_period: int = 12
    macd_slow_period: int = 26
    macd_signal_period: int = 9

    bbands_period: int = 20
    bbands_num_std: float = 2.0

    atr_period: int = 14

    # Longest lookback among the above, used to compute how much buffer (data history even further back than the queried window so that rolling indicators can be calculated without NaN's)
    warmup_buffer_days: int = 250  # comfortably > sma_long_period (200)


# ---------------------------------------------------------------------------
# Data / training universe / windows
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DataConfig:
    """
    DataConfig contains the list of hardcoded tickers representing stocks that will be used to train the NN.
    The training dataset will be independent from whichever single stock is queried for experimentation.
    """
    # List of tickers for training the NN,
    training_tickers: tuple = (
        "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AVGO", "ORCL", "CRM",
        "JPM", "BAC", "WFC", "GS", "MS", "C", "AXP", "SCHW", "BLK", "SPGI",
        "JNJ", "PFE", "MRK", "ABBV", "LLY", "UNH", "TMO", "ABT", "BMY", "MDT",
        "XOM", "CVX", "COP", "SLB", "EOG",
        "PG", "KO", "PEP", "WMT", "COST", "MCD", "NKE", "HD", "LOW", "TGT",
        "CAT", "BA", "GE", "HON", "UPS", "LMT", "RTX",
        "DIS", "NFLX", "CMCSA", "T", "VZ",
        "LIN", "NEE", "DUK", "SO",
        "PYPL", "ADBE", "CSCO", "INTC", "AMD", "QCOM", "TXN", "IBM",
    )

    # Number of years of daily history to pull PER TICKER when building the
    # training dataset (distinct from the single-stock query window)
    training_history_years: int = 8

    # Train/validation split strategy: "ticker" holds out entire tickers for
    # validation, "time" holds out a trailing time slice across all tickers.
    train_val_split_method: str = "ticker"  # "ticker" | "time"
    train_val_split_fraction: float = 0.2  # fraction held out for validation

    # Experiment-time window (single stock, per-experiment).
    experiment_history_days: int = 365  # ~1 year, window fed in at inference



@dataclass(frozen=True)
class LabelConfig:
    """
    The NN predicts forward volatility over a fixed horizon, which is a real quantity that can be computed from historical data.
    """
    forward_volatility_horizon_days: int = 21  # ~1 trading month ahead




@dataclass(frozen=True)
class ModelConfig:
    """
    Contains relevant information for NN architecture and training parameters.
    """
    hidden_layer_sizes: tuple = (64, 32, 16)
    activation: str = "relu"
    learning_rate: float = 1e-3
    epochs: int = 200
    batch_size: int = 32
    early_stopping_patience: int = 15
    save_path: str = "model/saved/volatility_predictor.keras"



@dataclass(frozen=True)
class CrossoverMappingConfig:
    """
    Configuration for mapping the NN's predicted forward volatility to a crossover rate.
    This is done via a hardcoded mappig function, to produce a crossover rate within the below specified range.
    """
    # specified range
    crossover_rate_min: float = 0.6
    crossover_rate_max: float = 0.9

    # Reference volatility range used to normalize the NN's raw predicted
    # volatility before mapping it linearly into
    # [crossover_rate_min, crossover_rate_max]. Should perform sanity-check these against
    # the real distribution of forward volatility once the training set is built
    volatility_reference_min: float = 0.10  # ~10% annualized, low-vol stock
    volatility_reference_max: float = 0.80  # ~80% annualized, high-vol stock




@dataclass(frozen=True)
class GeneSpec:
    """
    Constructing chromosome gene definitions for how a stock is represented in the GA.
    The definition contains its name, the data type, and its range of values to ensure logical validity.
    """
    name: str
    dtype: str  # "int" | "float"
    low: float
    high: float


@dataclass(frozen=True)
class RelationalConstraint:
    """
    Defining relational constraints between genes to ensure logical validity of the chromosome.
    """
    left: str
    op: str  # "<" | "<=" | ">" | ">="
    right: str


@dataclass(frozen=True)
class ChromosomeConfig:
    genes: tuple = (
        GeneSpec("sma_short_window",          "int",   5,    50),
        GeneSpec("sma_long_window",           "int",   50,   200),
        GeneSpec("rsi_period",                "int",   5,    30),
        GeneSpec("rsi_oversold",              "float", 10,   40),
        GeneSpec("rsi_overbought",            "float", 60,   90),
        GeneSpec("macd_fast_period",          "int",   5,    20),
        GeneSpec("macd_slow_period",          "int",   20,   40),
        GeneSpec("macd_signal_period",        "int",   5,    15),
        GeneSpec("bbands_window",             "int",   10,   40),
        GeneSpec("bbands_num_std",            "float", 1.0,  3.0),
        GeneSpec("stop_loss_pct",             "float", 0.01, 0.20),
        GeneSpec("take_profit_pct",           "float", 0.02, 0.50),
        GeneSpec("position_size_pct",         "float", 0.05, 1.0),
        GeneSpec("signal_combination_weight", "float", 0.0,  1.0),
    )

    constraints: tuple = (
        RelationalConstraint("sma_short_window", "<", "sma_long_window"),
        RelationalConstraint("macd_fast_period", "<", "macd_slow_period"),
        RelationalConstraint("rsi_oversold", "<", "rsi_overbought"),
    )



@dataclass(frozen=True)
class GAConfig:
    """
    Configuration class for the Genetic Algorithm, allowing for parameters to be changed.
    """
    # Population
    population_size: int = 100  # Size of the population
    max_gen: int = 300   # Number of generations to run
    # Operators (Crossover rate not included, as it will be predicted per stock)
    crossover: str = "OX"  # Crossover operator to use: "UOX" or another yet still to be decided
    mutation_rate: float = 0.2  # Probability of mutation
    # Selection
    tournament_k: int = 3   # Number of individuals to select for tournament selection
    # Elitism
    elitism: int = 10  # Number of the top chromsomes to carry over to the next generation

    def display_ga_config(self):
        """
        Display the current configuration settings for GA.
        """
        print("Configuration Settings:")
        print(f"Population Size: {self.popSize}")
        print(f"Max Generations: {self.max_gen}")
        print(f"Crossover Operator: {self.crossover}")
        print(f"Crossover Rate: {self.crossover_rate}")
        print(f"Mutation Rate: {self.mutation_rate}")
        print(f"Tournament Selection k: {self.tournament_k}")
        print(f"Elitism Count: {self.elitism}")
        print(f"Random Seed: {self.random_seed}")


@dataclass(frozen=True)
class Config:
    """
    Now combining all the above configuration classes into a single Config class, which can be imported and used throughout the project.
    """
    indicators: IndicatorConfig = field(default_factory=IndicatorConfig)
    data: DataConfig = field(default_factory=DataConfig)
    labels: LabelConfig = field(default_factory=LabelConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    crossover_mapping: CrossoverMappingConfig = field(default_factory=CrossoverMappingConfig)
    chromosome: ChromosomeConfig = field(default_factory=ChromosomeConfig)
    ga: GAConfig = field(default_factory=GAConfig)


# Single shared instance, so this can be imported everywhere
CONFIG = Config()