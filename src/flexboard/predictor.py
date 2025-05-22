"""Simplified performance predictor for MLPerf configurations using XGBoost."""

import logging
import random
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from utils import FEATURE_TYPES

logger = logging.getLogger(__name__)


class PerformancePredictor:
    """Predicts performance for hardware configurations."""

    def __init__(self, dataset: pd.DataFrame, test_size: float = 0.2):
        """Initialize with benchmark dataset."""
        self.df = dataset
        self.model = None
        self.target = "metrics.result_per_accelerator"
        self.features = []
        self.test_size = test_size

        self.evaluation_data = pd.DataFrame()
        self.evaluation_metrics = {}
        self.feature_importance = pd.DataFrame(columns=["Feature", "Importance"])

        self.excluded_features = {
            "model.name",
            "model.mlperf_name",
            "software.framework",
            "system.name",
        }

        self.excluded_features.update(
            {
                col
                for col in dataset.columns
                if col.startswith("submission.") or col.startswith("metrics.")
            }
        )

        self.distributions = {}

        self.max_accelerators = int(dataset["system.accelerator.total_count"].max())
        self.max_gpu_memory = float(dataset["system.accelerator.memory_capacity"].max())
        self.max_cpu_memory = float(dataset["system.memory.capacity"].max())

        self.frameworks = sorted(
            list(
                set(
                    col.replace("software.framework.", "")
                    for col in dataset.columns
                    if col.startswith("software.framework.")
                    and col != "software.framework"
                )
            )
        )
        logger.info(
            f"Found {len(self.frameworks)} unique frameworks: {', '.join(self.frameworks)}"
        )

        self._identify_features()
        self._analyze_data_distributions()
        self._train_model()

    def _identify_features(self):
        """Identify features for model training."""
        all_columns = set(self.df.columns)
        available_features = all_columns - self.excluded_features - {self.target}
        self.features = [f for f in available_features if not self.df[f].isna().all()]
        logger.info(f"Identified {len(self.features)} features for model training")

    def _analyze_data_distributions(self):
        """Analyze feature distributions for realistic data generation."""
        categorical_features = {
            col
            for col in self.df.columns
            if self.df[col].dtype == "object"
            or col in FEATURE_TYPES.get("categorical", [])
        }

        for feature in categorical_features:
            values = self.df[feature].dropna().tolist()
            if values:
                counter = Counter(values)
                total = sum(counter.values())
                self.distributions[feature] = {
                    value: count / total for value, count in counter.items()
                }

        continuous_features = {
            col
            for col in self.df.columns
            if col in FEATURE_TYPES.get("continuous", [])
            or pd.api.types.is_numeric_dtype(self.df[col].dtype)
            if col not in categorical_features and not col.startswith("metrics.")
        }

        for feature in continuous_features:
            values = self.df[feature].dropna()
            if len(values) > 0:
                self.distributions[feature] = {
                    "min": float(values.min()),
                    "max": float(values.max()),
                    "mean": float(values.mean()),
                    "std": float(values.std()),
                    "median": float(values.median()),
                    "values": sorted(values.unique().tolist()),
                }

        self._analyze_feature_relationships()
        logger.info(f"Analyzed distributions for {len(self.distributions)} features")

    def _analyze_feature_relationships(self):
        """Analyze relationships between related features."""
        self._analyze_vendor_accelerator_relations()
        self._analyze_vendor_cpu_relations()
        self._analyze_accelerator_memory_relations()
        self._analyze_interconnect_relations()
        self._analyze_software_relations()
        self._analyze_device_counts()

    def _analyze_vendor_accelerator_relations(self):
        """Map vendors to their accelerators."""
        vendor_accelerators = defaultdict(list)
        for _, row in self.df.iterrows():
            vendor = row.get("system.accelerator.vendor")
            acc = row.get("system.accelerator.name")
            if vendor and acc:
                vendor_accelerators[vendor].append(acc)

        self.distributions["vendor_accelerators"] = {}
        for vendor, accelerators in vendor_accelerators.items():
            counter = Counter(accelerators)
            total = sum(counter.values())
            self.distributions["vendor_accelerators"][vendor] = {
                acc: count / total for acc, count in counter.items()
            }

    def _analyze_vendor_cpu_relations(self):
        """Map CPU vendors to their models."""
        vendor_cpus = defaultdict(list)
        for _, row in self.df.iterrows():
            vendor = row.get("system.cpu.vendor")
            model = row.get("system.cpu.model")
            if vendor and model:
                vendor_cpus[vendor].append(model)

        self.distributions["vendor_cpus"] = {}
        for vendor, models in vendor_cpus.items():
            counter = Counter(models)
            total = sum(counter.values())
            self.distributions["vendor_cpus"][vendor] = {
                model: count / total for model, count in counter.items()
            }

    def _analyze_accelerator_memory_relations(self):
        """Map accelerator models to memory capacities."""
        acc_memory = defaultdict(list)
        for _, row in self.df.iterrows():
            acc = row.get("system.accelerator.name")
            memory = row.get("system.accelerator.memory_capacity")
            if acc and memory:
                acc_memory[acc].append(memory)

        self.distributions["accelerator_memory"] = {}
        for acc, memories in acc_memory.items():
            if memories:
                counter = Counter(memories)
                most_common = counter.most_common(1)[0][0] if counter else None
                self.distributions["accelerator_memory"][acc] = {
                    "min": min(memories),
                    "max": max(memories),
                    "mean": sum(memories) / len(memories),
                    "most_common": most_common,
                    "values": sorted(set(memories)),
                }

    def _analyze_interconnect_relations(self):
        """Map vendors to interconnect types."""
        vendor_interconnects = defaultdict(list)
        for _, row in self.df.iterrows():
            vendor = row.get("system.accelerator.vendor")
            interconnect = row.get("system.interconnect.accelerator")
            if vendor and interconnect:
                vendor_interconnects[vendor].append(interconnect)

        self.distributions["vendor_interconnects"] = {}
        for vendor, interconnects in vendor_interconnects.items():
            counter = Counter(interconnects)
            total = sum(counter.values())
            self.distributions["vendor_interconnects"][vendor] = {
                ic: count / total for ic, count in counter.items()
            }

    def _analyze_software_relations(self):
        """Map vendors to software stacks."""
        vendor_software = defaultdict(lambda: defaultdict(list))
        for _, row in self.df.iterrows():
            vendor = row.get("system.accelerator.vendor")
            if not vendor:
                continue

            os = row.get("software.operating_system")
            if os:
                vendor_software[vendor]["os"].append(os)

            for col in self.df.columns:
                if (
                    col.startswith("software.framework.")
                    and col != "software.framework"
                ):
                    framework = col.replace("software.framework.", "")
                    version = row.get(col)
                    if version:
                        vendor_software[vendor][framework].append(version)

        self.distributions["vendor_software"] = {}
        for vendor, software_dict in vendor_software.items():
            self.distributions["vendor_software"][vendor] = {}
            for software_type, values in software_dict.items():
                counter = Counter(values)
                total = sum(counter.values())
                self.distributions["vendor_software"][vendor][software_type] = {
                    value: count / total for value, count in counter.items()
                }

    def _analyze_device_counts(self):
        """Analyze distribution of device counts."""
        counts = self.df["system.accelerator.total_count"].dropna().astype(int).tolist()
        if counts:
            counter = Counter(counts)
            total = sum(counter.values())
            self.distributions["device_count"] = {
                count: freq / total for count, freq in counter.items()
            }
            self.distributions["device_count_values"] = sorted(list(set(counts)))

        node_counts = self.df["system.number_of_nodes"].dropna().astype(int).tolist()
        if node_counts:
            counter = Counter(node_counts)
            total = sum(counter.values())
            self.distributions["node_count"] = {
                count: freq / total for count, freq in counter.items()
            }
            self.distributions["node_count_values"] = sorted(list(set(node_counts)))

        device_node_pairs = [
            (
                int(row["system.number_of_nodes"]),
                int(row["system.accelerator.total_count"]),
            )
            for _, row in self.df.iterrows()
            if row.get("system.number_of_nodes")
            and row.get("system.accelerator.total_count")
        ]

        node_to_devices = defaultdict(list)
        for nodes, devices in device_node_pairs:
            node_to_devices[nodes].append(devices)

        self.distributions["node_device_relation"] = {}
        for node_count, device_counts in node_to_devices.items():
            counter = Counter(device_counts)
            total = sum(counter.values())
            self.distributions["node_device_relation"][node_count] = {
                count: freq / total for count, freq in counter.items()
            }

    def _train_model(self):
        """Train XGBoost model on available data with train/test split."""
        df_clean = self.df.dropna(subset=[self.target])

        X = df_clean[self.features]
        y = df_clean[self.target]

        for col in X.select_dtypes(include=["object"]).columns:
            with pd.option_context("mode.chained_assignment", None):
                X[col] = X[col].astype("category")

        try:
            strat_column = df_clean["system.accelerator.name"].fillna("unknown")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size, stratify=strat_column, random_state=42
            )
            logger.info(
                f"Created stratified train/test split ({100 - self.test_size * 100:.0f}%/{self.test_size * 100:.0f}%) with {len(X_train)} training and {len(X_test)} test samples"
            )
        except ValueError:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size, random_state=42
            )
            logger.info(
                f"Created regular train/test split with {len(X_train)} training and {len(X_test)} test samples"
            )

        self.model = xgb.XGBRegressor(
            objective="reg:squarederror",
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            enable_categorical=True,
        )

        self.model.fit(X_train, y_train)
        logger.info(f"Trained XGBoost model on {len(X_train)} rows")

        self._evaluate_model(X_test, y_test, df_clean.loc[X_test.index])

    def _evaluate_model(self, X_test, y_test, test_df):
        """Evaluate model performance on test set."""
        if X_test.empty:
            logger.warning("No test data available for evaluation")
            return

        y_pred = self.model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

        self.evaluation_metrics = {
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "mape": mape,
            "test_size": len(y_test),
            "training_size": len(self.df) - len(y_test),
        }

        eval_data = test_df[
            [
                "system.accelerator.name",
                "system.accelerator.vendor",
                "system.accelerator.total_count",
            ]
        ].copy()
        eval_data["actual"] = y_test
        eval_data["predicted"] = y_pred
        eval_data["error"] = y_pred - y_test
        eval_data["error_percent"] = (y_pred - y_test) / y_test * 100

        self.evaluation_data = eval_data.copy()

        logger.info(
            f"Model evaluation - RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.3f}, MAPE: {mape:.2f}%"
        )
        logger.info(
            f"Evaluation data shape: {eval_data.shape}, with columns: {list(eval_data.columns)}"
        )
        logger.info(f"Evaluation data sample: {eval_data.head(2).to_dict()}")
        logger.info(
            f"Evaluation data stored as class attribute with shape: {self.evaluation_data.shape}"
        )

        importance = self.model.feature_importances_
        feature_importance = pd.DataFrame(
            {"Feature": self.model.feature_names_in_, "Importance": importance}
        ).sort_values("Importance", ascending=False)
        self.feature_importance = feature_importance.head(10).copy()

        logger.info(
            f"Top 5 important features: {', '.join(self.feature_importance['Feature'].head(5).tolist())}"
        )

    def get_evaluation_metrics(self) -> dict:
        """Return model evaluation metrics."""
        logger.info(f"Getting evaluation metrics: {self.evaluation_metrics}")
        return self.evaluation_metrics.copy() if self.evaluation_metrics else {}

    def get_evaluation_data(self) -> pd.DataFrame:
        """Return evaluation data for visualization."""
        data_shape = (
            "empty" if self.evaluation_data.empty else self.evaluation_data.shape
        )
        logger.info(f"Getting evaluation data with shape: {data_shape}")
        return self.evaluation_data.copy() if not self.evaluation_data.empty else None

    def get_feature_importance(self) -> pd.DataFrame:
        """Return feature importance data."""
        logger.info(
            f"Getting feature importance with shape: {self.feature_importance.shape}"
        )
        return (
            self.feature_importance.copy()
            if not self.feature_importance.empty
            else pd.DataFrame(columns=["Feature", "Importance"])
        )

    def generate_predictions(
        self,
        architecture: str,
        parameters: float,
        constraints: dict = None,
        num_configs: int = 10,
    ) -> pd.DataFrame:
        """Generate and predict performance for hardware configurations."""
        constraints = constraints or {}
        logger.info(
            f"Generating {num_configs} predictions for {architecture} model with {parameters}B parameters"
        )

        configs = self._generate_configs(
            architecture, parameters, constraints, num_configs
        )
        if not configs:
            return pd.DataFrame()

        configs_df = pd.DataFrame(configs)
        model_features = getattr(self.model, "feature_names_in_", self.features)

        for feature in model_features:
            if feature not in configs_df.columns:
                configs_df[feature] = None

        X_pred = configs_df[model_features]
        for col in X_pred.select_dtypes(include=["object"]).columns:
            with pd.option_context("mode.chained_assignment", None):
                X_pred[col] = X_pred[col].astype("category")

        configs_df[self.target] = self.model.predict(X_pred)
        configs_df["predicted"] = True
        configs_df["metrics.result"] = (
            configs_df[self.target] * configs_df["system.accelerator.total_count"]
        )
        configs_df["system.name"] = "Hypothetical system - ongoing work"

        logger.info(
            f"Performance range: {configs_df[self.target].min():.2f} - {configs_df[self.target].max():.2f} tokens/s per accelerator"
        )
        return configs_df

    def _sample_from_distribution(self, distribution: dict) -> any:
        """Sample a value from a categorical distribution."""
        items = list(distribution.keys())
        probabilities = list(distribution.values())
        return np.random.choice(items, p=probabilities)

    def _sample_continuous_value(self, feature: str) -> float:
        """Sample a continuous value from the feature distribution."""
        dist = self.distributions[feature]

        if "values" in dist and dist["values"]:
            if len(dist["values"]) > 3:
                value = np.random.normal(dist["mean"], max(dist["std"], 1.0))
                value = max(dist["min"], min(dist["max"], value))
                closest_idx = min(
                    range(len(dist["values"])),
                    key=lambda i: abs(dist["values"][i] - value),
                )
                return dist["values"][closest_idx]
            else:
                return random.choice(dist["values"])

        elif all(k in dist for k in ["min", "max", "mean", "std"]):
            value = np.random.normal(dist["mean"], max(dist["std"], 1.0))
            return max(dist["min"], min(dist["max"], value))

        return np.random.uniform(dist["min"], dist["max"])

    def _get_device_count(self, min_devices=None, max_devices=None) -> int:
        """Get a realistic device count based on distribution and constraints."""
        valid_counts = [
            count
            for count in self.distributions["device_count_values"]
            if (min_devices is None or count >= min_devices)
            and (max_devices is None or count <= max_devices)
        ]

        if valid_counts:
            probs = {
                count: self.distributions["device_count"][count]
                for count in valid_counts
                if count in self.distributions["device_count"]
            }

            if probs:
                total = sum(probs.values())
                items = list(probs.keys())
                weights = [probs[item] / total for item in items]
                return np.random.choice(items, p=weights)

            return random.choice(valid_counts)

        if min_devices is not None and max_devices is not None:
            valid_powers = [
                2**i for i in range(10) if min_devices <= 2**i <= max_devices
            ]
            if valid_powers:
                return random.choice(valid_powers)
            return random.randint(min_devices, max_devices)

        return random.choice([1, 2, 4, 8, 16])

    def _get_vendor_accelerator(self, vendor=None) -> tuple:
        """Get a vendor and accelerator pair."""
        if vendor is None or vendor == "Any":
            vendor = self._sample_from_distribution(
                self.distributions["system.accelerator.vendor"]
            )

        if vendor in self.distributions["vendor_accelerators"]:
            accelerator = self._sample_from_distribution(
                self.distributions["vendor_accelerators"][vendor]
            )
        else:
            accelerator = self._sample_from_distribution(
                self.distributions["system.accelerator.name"]
            )

        return vendor, accelerator

    def _get_memory_for_accelerator(
        self, vendor: str, accelerator: str, min_memory=None, max_memory=None
    ) -> float:
        """Get appropriate memory capacity for a given accelerator."""
        if accelerator in self.distributions["accelerator_memory"]:
            mem_dist = self.distributions["accelerator_memory"][accelerator]

            if "values" in mem_dist:
                valid_values = [
                    m
                    for m in mem_dist["values"]
                    if (min_memory is None or m >= min_memory)
                    and (max_memory is None or m <= max_memory)
                ]
                if valid_values:
                    return random.choice(valid_values)

            if "most_common" in mem_dist:
                most_common = mem_dist["most_common"]
                if (min_memory is None or most_common >= min_memory) and (
                    max_memory is None or most_common <= max_memory
                ):
                    return most_common

        dist = self.distributions["system.accelerator.memory_capacity"]
        valid_values = [
            m
            for m in dist["values"]
            if (min_memory is None or m >= min_memory)
            and (max_memory is None or m <= max_memory)
        ]

        if valid_values:
            return random.choice(valid_values)

        min_val = max(dist["min"], min_memory or dist["min"])
        max_val = min(dist["max"], max_memory or dist["max"])

        if min_val <= max_val:
            mean = min(max(dist["mean"], min_val), max_val)
            std = max(dist["std"], 1.0)

            for _ in range(5):
                value = np.random.normal(mean, std)
                if min_val <= value <= max_val:
                    return value

            return np.random.uniform(min_val, max_val)

        return None

    def _get_node_config(self, total_devices: int) -> tuple:
        """Determine number of nodes and devices per node."""
        VALID_GPUS_PER_NODE = [1, 2, 4, 8]

        for gpus_per_node in sorted(VALID_GPUS_PER_NODE, reverse=True):
            if total_devices % gpus_per_node == 0:
                return total_devices // gpus_per_node, gpus_per_node

        for gpus_per_node in sorted(VALID_GPUS_PER_NODE, reverse=True):
            if gpus_per_node <= total_devices:
                nodes = total_devices // gpus_per_node
                return nodes, gpus_per_node

        return 1, 1

    def _get_cpu_config(self) -> dict:
        """Generate a CPU configuration."""
        cpu_config = {}
        cpu_config["system.cpu.vendor"] = self._sample_from_distribution(
            self.distributions["system.cpu.vendor"]
        )

        cpu_vendor = cpu_config["system.cpu.vendor"]
        if cpu_vendor in self.distributions["vendor_cpus"]:
            cpu_config["system.cpu.model"] = self._sample_from_distribution(
                self.distributions["vendor_cpus"][cpu_vendor]
            )
        else:
            cpu_config["system.cpu.model"] = self._sample_from_distribution(
                self.distributions["system.cpu.model"]
            )

        for feature in [
            "system.cpu.core_count",
            "system.cpu.count_per_node",
            "system.cpu.frequency",
        ]:
            value = self._sample_continuous_value(feature)
            if value is not None:
                if feature in ["system.cpu.core_count", "system.cpu.count_per_node"]:
                    value = int(value)
                cpu_config[feature] = value

        if "system.cpu.caches" in self.distributions:
            cpu_config["system.cpu.caches"] = self._sample_from_distribution(
                self.distributions["system.cpu.caches"]
            )

        return cpu_config

    def _get_software_config(self, vendor: str, constraints=None) -> dict:
        """Generate a software configuration based on hardware vendor."""
        constraints = constraints or {}
        software_config = {}

        if vendor in self.distributions["vendor_software"]:
            vendor_sw = self.distributions["vendor_software"][vendor]

            if "os" in vendor_sw:
                os_constraint = constraints.get("software.operating_system")
                if os_constraint and os_constraint != "Any":
                    software_config["software.operating_system"] = os_constraint
                else:
                    software_config["software.operating_system"] = (
                        self._sample_from_distribution(vendor_sw["os"])
                    )

            for framework, versions in vendor_sw.items():
                if framework != "os":
                    framework_key = f"software.framework.{framework}"
                    version_constraint = constraints.get(framework_key)
                    if version_constraint and version_constraint != "Any":
                        software_config[framework_key] = version_constraint
                    else:
                        software_config[framework_key] = self._sample_from_distribution(
                            versions
                        )

        if (
            "software.operating_system" not in software_config
            and "software.operating_system" in self.distributions
        ):
            os_constraint = constraints.get("software.operating_system")
            if os_constraint and os_constraint != "Any":
                software_config["software.operating_system"] = os_constraint
            else:
                software_config["software.operating_system"] = (
                    self._sample_from_distribution(
                        self.distributions["software.operating_system"]
                    )
                )

        for framework in self.frameworks:
            framework_key = f"software.framework.{framework}"
            if (
                framework_key not in software_config
                and framework_key in self.distributions
            ):
                version_constraint = constraints.get(framework_key)
                if version_constraint and version_constraint != "Any":
                    software_config[framework_key] = version_constraint
                else:
                    software_config[framework_key] = self._sample_from_distribution(
                        self.distributions[framework_key]
                    )

        return software_config

    def _get_memory_config(self, min_memory=None, max_memory=None) -> dict:
        """Generate a memory configuration."""
        memory_config = {}
        dist = self.distributions["system.memory.capacity"]

        if "values" in dist:
            valid_values = [
                m
                for m in dist["values"]
                if (min_memory is None or m >= min_memory)
                and (max_memory is None or m <= max_memory)
            ]
            if valid_values:
                memory_config["system.memory.capacity"] = random.choice(valid_values)

        if "system.memory.capacity" not in memory_config:
            min_val = max(dist["min"], min_memory or dist["min"])
            max_val = min(dist["max"], max_memory or dist["max"])

            if min_val <= max_val:
                mean = min(max(dist["mean"], min_val), max_val)
                std = max(dist["std"], (max_val - min_val) / 6.0)

                value = np.random.normal(mean, std)
                if min_val <= value <= max_val:
                    memory_config["system.memory.capacity"] = value
                else:
                    memory_config["system.memory.capacity"] = np.random.uniform(
                        min_val, max_val
                    )

        if "system.memory.configuration" in self.distributions:
            memory_config["system.memory.configuration"] = (
                self._sample_from_distribution(
                    self.distributions["system.memory.configuration"]
                )
            )

        return memory_config

    def _get_interconnect_config(self, vendor: str) -> dict:
        """Generate interconnect configuration based on vendor."""
        interconnect_config = {}

        if vendor in self.distributions["vendor_interconnects"]:
            interconnect_config["system.interconnect.accelerator"] = (
                self._sample_from_distribution(
                    self.distributions["vendor_interconnects"][vendor]
                )
            )
        elif "system.interconnect.accelerator" in self.distributions:
            interconnect_config["system.interconnect.accelerator"] = (
                self._sample_from_distribution(
                    self.distributions["system.interconnect.accelerator"]
                )
            )

        if "system.interconnect.accelerator_host" in self.distributions:
            interconnect_config["system.interconnect.accelerator_host"] = (
                self._sample_from_distribution(
                    self.distributions["system.interconnect.accelerator_host"]
                )
            )

        return interconnect_config

    def _generate_configs(
        self, architecture: str, parameters: float, constraints=None, count: int = 10
    ) -> list:
        """Generate configurations respecting user constraints."""
        constraints = constraints or {}
        configs = []

        vendor = constraints.get("system.accelerator.vendor")
        acc_name = constraints.get("system.accelerator.name")

        def apply_margin(value, is_min=True):
            if value is None or not isinstance(value, (int, float)) or value == "Any":
                return None
            return value * (0.9 if is_min else 1.1)

        min_gpu_memory = apply_margin(constraints.get("min_gpu_memory"), is_min=True)
        max_gpu_memory = apply_margin(
            constraints.get("max_gpu_memory"), is_min=False
        ) or (self.max_gpu_memory * 1.1)

        min_cpu_memory = apply_margin(constraints.get("min_cpu_memory"), is_min=True)
        max_cpu_memory = apply_margin(
            constraints.get("max_cpu_memory"), is_min=False
        ) or (self.max_cpu_memory * 1.1)

        min_devices = apply_margin(constraints.get("min_accelerators"), is_min=True)
        max_devices = (
            apply_margin(constraints.get("max_accelerators"), is_min=False)
            or self.max_accelerators
        )

        interconnect = constraints.get("system.interconnect.accelerator")
        nodes = constraints.get("system.number_of_nodes")

        VALID_GPUS_PER_NODE = [1, 2, 4, 8]

        for _ in range(count * 3):
            if len(configs) >= count:
                break

            device_count = self._get_device_count(min_devices, max_devices)
            acc_vendor, acc_model = self._get_vendor_accelerator(vendor)

            if acc_name and acc_name != "Any":
                acc_model = acc_name

            if nodes and nodes != "Any":
                node_count = int(nodes)
                valid_device_counts = []
                for gpus in VALID_GPUS_PER_NODE:
                    if node_count * gpus >= (
                        min_devices or 1
                    ) and node_count * gpus <= (max_devices or float("inf")):
                        valid_device_counts.append(gpus)

                if not valid_device_counts:
                    continue

                devices_per_node = random.choice(valid_device_counts)
                device_count = node_count * devices_per_node
            else:
                valid_count = False
                for gpus_per_node in sorted(VALID_GPUS_PER_NODE, reverse=True):
                    if device_count % gpus_per_node == 0:
                        valid_count = True
                        break

                if not valid_count:
                    node_count, devices_per_node = self._get_node_config(device_count)
                    device_count = node_count * devices_per_node
                else:
                    node_count, devices_per_node = (
                        device_count // gpus_per_node,
                        gpus_per_node,
                    )

            config = {
                "model.architecture": architecture,
                "model.number_of_parameters": parameters,
                "system.accelerator.vendor": acc_vendor,
                "system.accelerator.name": acc_model,
                "system.accelerator.total_count": device_count,
                "system.number_of_nodes": node_count,
                "system.accelerator.count_per_node": devices_per_node,
            }

            gpu_memory = self._get_memory_for_accelerator(
                acc_vendor,
                acc_model,
                min_memory=min_gpu_memory,
                max_memory=max_gpu_memory,
            )

            if gpu_memory is None:
                continue

            config["system.accelerator.memory_capacity"] = gpu_memory

            if "system.accelerator.memory_config" in self.distributions:
                config["system.accelerator.memory_config"] = (
                    self._sample_from_distribution(
                        self.distributions["system.accelerator.memory_config"]
                    )
                )

            interconnect_config = self._get_interconnect_config(acc_vendor)

            if interconnect and interconnect != "Any":
                interconnect_config["system.interconnect.accelerator"] = interconnect

            config.update(interconnect_config)
            config.update(self._get_cpu_config())

            memory_config = self._get_memory_config(
                min_memory=min_cpu_memory, max_memory=max_cpu_memory
            )
            if "system.memory.capacity" not in memory_config:
                continue

            config.update(memory_config)

            for feature_name in [
                "system.type",
                "system.cooling",
                "model.weight_data_types",
            ]:
                if feature_name in self.distributions:
                    config[feature_name] = self._sample_from_distribution(
                        self.distributions[feature_name]
                    )

            config.update(self._get_software_config(acc_vendor, constraints))

            for key, value in constraints.items():
                if (
                    not key.startswith("software.framework.")
                    and key != "software.operating_system"
                    and key
                    not in [
                        "min_gpu_memory",
                        "max_gpu_memory",
                        "min_cpu_memory",
                        "max_cpu_memory",
                        "min_accelerators",
                        "max_accelerators",
                    ]
                    and key not in config
                    and value != "Any"
                    and value is not None
                ):
                    config[key] = value

            configs.append(config)

        return configs[:count]
