import pandas as pd
import numpy as np
import logging
import joblib
from typing import Tuple, Dict, List, Optional
from pathlib import Path

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import NotFittedError

import xgboost as xgb
import lightgbm as lgb
import catboost as cb


class EnsemblePredictor:
    def __init__(self):
        self.models = {
            "xgboost": CalibratedClassifierCV(
                xgb.XGBClassifier(
                    objective='binary:logistic',
                    n_estimators=200,
                    learning_rate=0.1,
                    max_depth=5,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    eval_metric='logloss'
                ),
                method='isotonic',
                cv=5
            ),
            "lightgbm": CalibratedClassifierCV(
                lgb.LGBMClassifier(
                    n_estimators=200,
                    learning_rate=0.1,
                    max_depth=5,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42
                ),
                method='isotonic',
                cv=5
            ),
            "catboost": CalibratedClassifierCV(
                cb.CatBoostClassifier(
                    iterations=200,
                    learning_rate=0.1,
                    depth=5,
                    subsample=0.8,
                    colsample_bylevel=0.8,
                    random_seed=42,
                    verbose=False
                ),
                method='isotonic',
                cv=5
            ),
            "random_forest": RandomForestClassifier(
                n_estimators=200,
                max_depth=5,
                random_state=42
            )
        }
        
        self.ensemble = VotingClassifier(
            estimators=[(name, model) for name, model in self.models.items()],
            voting='soft',
            weights=[1.2, 1.1, 1.1, 1.0]
        )
        self.feature_columns = None
        self.scaler = StandardScaler()
        self.held_out_matches = None
        self.feature_importance = None
        
    def tune_hyperparameters(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict:
        try:
            param_grid = {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'min_child_weight': [1, 3, 5],
                'subsample': [0.6, 0.8, 1.0],
                'colsample_bytree': [0.6, 0.8, 1.0]
            }
            
            xgb_model = xgb.XGBClassifier(
                objective='binary:logistic',
                random_state=42,
                eval_metric='logloss'
            )
            
            grid_search = GridSearchCV(
                estimator=xgb_model,
                param_grid=param_grid,
                scoring='neg_log_loss',
                cv=3,
                n_jobs=-1
            )
            
            grid_search.fit(X_train, y_train)
            best_params = grid_search.best_params_
            
            logging.info("Best hyperparameters:")
            for param, value in best_params.items():
                logging.info(f"  {param}: {value}")
            
            self.models["xgboost"] = CalibratedClassifierCV(
                xgb.XGBClassifier(
                    objective='binary:logistic',
                    random_state=42,
                    eval_metric='logloss',
                    **best_params
                ),
                method='isotonic',
                cv=5
            )
            
            self.ensemble = VotingClassifier(
                estimators=[(name, model) for name, model in self.models.items()],
                voting='soft'
            )
            
            return best_params
            
        except Exception as e:
            logging.error(f"Error tuning hyperparameters: {e}")
            raise
        
    def _create_interaction_features(self, X: pd.DataFrame) -> pd.DataFrame:
        interaction_features = pd.DataFrame(index=X.index)
        
        for stat in ['avg_rating', 'top2_rating', 'avg_adr', 'avg_kd_ratio']:
            team1_stat = f'team1_{stat}'
            team2_stat = f'team2_{stat}'
            if team1_stat in X.columns and team2_stat in X.columns:
                interaction_features[f'{stat}_ratio'] = X[team1_stat] / (X[team2_stat] + 1e-6)
                interaction_features[f'{stat}_diff'] = X[team1_stat] - X[team2_stat]
                interaction_features[f'{stat}_product'] = X[team1_stat] * X[team2_stat]
        
        for stat in ['avg_rating', 'top2_rating']:
            team1_stat = f'team1_{stat}'
            team2_stat = f'team2_{stat}'
            if team1_stat in X.columns and team2_stat in X.columns:
                interaction_features[f'{team1_stat}_squared'] = X[team1_stat] ** 2
                interaction_features[f'{team2_stat}_squared'] = X[team2_stat] ** 2
        
        return pd.concat([X, interaction_features], axis=1)
        
    def _preprocess_features(self, X: pd.DataFrame, is_training: bool = False) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
            
        original_columns = X.columns.tolist()
            
        if 'scrape_date' in X.columns:
            X['scrape_date'] = pd.to_datetime(X['scrape_date'])
            X['year'] = X['scrape_date'].dt.year
            X['month'] = X['scrape_date'].dt.month
            X['day'] = X['scrape_date'].dt.day
            X = X.drop(columns=['scrape_date'])
        
        if 'bo_type' not in X.columns:
            X['bo_type'] = 3
        X['bo_type'] = pd.to_numeric(X['bo_type'], errors='coerce').fillna(3)
        
        X = self._create_interaction_features(X)
        
        X_numeric = X.select_dtypes(include=['number'])
        
        if is_training:
            self.feature_columns = X_numeric.columns.tolist()
        
        if hasattr(self.scaler, 'mean_'):
            missing_cols = set(self.feature_columns) - set(X_numeric.columns)
            if missing_cols:
                for col in missing_cols:
                    X_numeric[col] = 0
            
            X_numeric = X_numeric.reindex(columns=self.feature_columns, fill_value=0)
            
            if hasattr(self.scaler, 'mean_'):
                X_numeric = pd.DataFrame(
                    self.scaler.transform(X_numeric),
                    columns=X_numeric.columns,
                    index=X_numeric.index
                )
            else:
                X_numeric = pd.DataFrame(
                    self.scaler.fit_transform(X_numeric),
                    columns=X_numeric.columns,
                    index=X_numeric.index
                )
        
        return X_numeric
        
    def train(self, X: pd.DataFrame, y: pd.Series, tune_hyperparams: bool = False) -> None:
        try:
            X_processed = self._preprocess_features(X, is_training=True)
            
            if len(X_processed) > 20:
                self.held_out_matches = {
                    'X': X_processed.tail(20),
                    'y': y.tail(20)
                }
                X_processed = X_processed.iloc[:-20]
                y = y.iloc[:-20]
            
            if tune_hyperparams:
                self.tune_hyperparameters(X_processed, y)
            
            for name, model in self.models.items():
                logging.info(f"Training {name} model...")
                model.fit(X_processed, y)
                if not hasattr(model, 'classes_'):
                    raise ValueError(f"{name} model was not properly fitted")
            
            logging.info("Training ensemble...")
            self.ensemble.fit(X_processed, y)
            
            self._calculate_feature_importance(X_processed, y)
            
            self._evaluate_model(X_processed, y)
            
            if self.held_out_matches is not None:
                logging.info("\nEvaluating on held-out data:")
                self._evaluate_model(
                    self.held_out_matches['X'],
                    self.held_out_matches['y']
                )
            
            for name, model in self.models.items():
                if not hasattr(model, 'classes_'):
                    raise ValueError(f"{name} model was not properly fitted after training")
            
            logging.info("All models successfully trained and fitted")
            
        except Exception as e:
            logging.error(f"Error training model: {e}")
            raise
        
    def _calculate_feature_importance(self, X: pd.DataFrame, y: pd.Series) -> None:
        try:
            xgb_importance = self.models['xgboost'].base_estimator.feature_importances_
            
            rf_importance = self.models['random_forest'].feature_importances_
            
            avg_importance = (xgb_importance + rf_importance) / 2
            
            self.feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': avg_importance
            }).sort_values('importance', ascending=False)
            
            logging.info("\nTop 10 most important features:")
            for _, row in self.feature_importance.head(10).iterrows():
                logging.info(f"  {row['feature']}: {row['importance']:.4f}")
                
        except Exception as e:
            logging.error(f"Error calculating feature importance: {e}")
        
    def _evaluate_model(self, X: pd.DataFrame, y: pd.Series) -> None:
        try:
            cv_scores = cross_val_score(
                self.ensemble,
                X,
                y,
                cv=5,
                scoring='neg_log_loss'
            )
            logging.info(f"Cross-validation log loss scores: {cv_scores}")
            logging.info(f"Mean CV log loss: {-cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
            
            y_pred = self.ensemble.predict(X)
            y_prob = self.ensemble.predict_proba(X)[:, 1]
            
            accuracy = accuracy_score(y, y_pred)
            brier = brier_score_loss(y, y_prob)
            logloss = log_loss(y, y_prob)
            roc_auc = roc_auc_score(y, y_prob)
            
            logging.info(f"Training set metrics:")
            logging.info(f"  Accuracy: {accuracy:.4f}")
            logging.info(f"  Brier score: {brier:.4f}")
            logging.info(f"  Log loss: {logloss:.4f}")
            logging.info(f"  ROC AUC: {roc_auc:.4f}")
            
            sample = pd.DataFrame({
                "winner_true": y.values,
                "winner_pred": y_pred,
                "prob_team1_win": y_prob
            })
            logging.info(f"\nSample predictions:\n{sample.head()}")
            
        except Exception as e:
            logging.error(f"Error evaluating model: {e}")
            raise
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not hasattr(self, 'feature_columns') or self.feature_columns is None:
            raise NotFittedError("Model has not been trained yet. Call train() first.")
            
        try:
            X_processed = self._preprocess_features(X, is_training=False)
            return self.ensemble.predict(X_processed)
        except Exception as e:
            logging.error(f"Error making predictions: {e}")
            raise
        
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not hasattr(self, 'feature_columns') or self.feature_columns is None:
            raise NotFittedError("Model has not been trained yet. Call train() first.")
            
        try:
            X_processed = self._preprocess_features(X, is_training=False)
            
            model_predictions = []
            for name, model in self.models.items():
                pred = model.predict_proba(X_processed)
                model_predictions.append(pred)
            
            weights = [1.2, 1.1, 1.1, 1.0]
            weighted_pred = np.average(model_predictions, weights=weights, axis=0)
            
            temperature = 1.2
            scaled_pred = weighted_pred ** (1/temperature)
            scaled_pred = scaled_pred / scaled_pred.sum(axis=1, keepdims=True)
            
            return scaled_pred
            
        except Exception as e:
            logging.error(f"Error making probability predictions: {e}")
            raise
        
    def save(self, model_path: str, feature_columns_path: str) -> None:
        try:
            Path(model_path).parent.mkdir(parents=True, exist_ok=True)
            
            for name, model in self.models.items():
                model_path_individual = str(Path(model_path).parent / f"{name}_model.joblib")
                joblib.dump(model, model_path_individual, compress=3)
            
            joblib.dump(self.ensemble, model_path, compress=3)
            joblib.dump(self.feature_columns, feature_columns_path)
            
            scaler_path = str(Path(model_path).parent / "scaler.joblib")
            joblib.dump(self.scaler, scaler_path)
            
            logging.info(f"Model saved to {model_path}")
            logging.info(f"Feature columns saved to {feature_columns_path}")
        except Exception as e:
            logging.error(f"Error saving model: {e}")
            raise
        
    @classmethod
    def load(cls, model_path: str, feature_columns_path: str) -> 'EnsemblePredictor':
        try:
            model = cls()
            
            for name in model.models.keys():
                model_path_individual = str(Path(model_path).parent / f"{name}_model.joblib")
                model.models[name] = joblib.load(model_path_individual)
            
            model.ensemble = joblib.load(model_path)
            model.feature_columns = joblib.load(feature_columns_path)
            
            scaler_path = str(Path(model_path).parent / "scaler.joblib")
            model.scaler = joblib.load(scaler_path)
            
            logging.info(f"Model loaded from {model_path}")
            return model
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            raise

class MatchWinPredictor:
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.model = None
        self.feature_columns = None
        self.held_out_matches = None

    def tune_hyperparameters(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict:
        try:
            param_grid = {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'min_child_weight': [1, 3, 5],
                'subsample': [0.6, 0.8, 1.0],
                'colsample_bytree': [0.6, 0.8, 1.0]
            }
            
            xgb_model = xgb.XGBClassifier(
                objective='binary:logistic',
                random_state=42,
                eval_metric='logloss'
            )
            
            grid_search = GridSearchCV(
                estimator=xgb_model,
                param_grid=param_grid,
                scoring='neg_log_loss',
                cv=3,
                n_jobs=-1
            )
            
            grid_search.fit(X_train, y_train)
            best_params = grid_search.best_params_
            
            logging.info("Best hyperparameters:")
            for param, value in best_params.items():
                logging.info(f"  {param}: {value}")
            
            return best_params
            
        except Exception as e:
            logging.error(f"Error tuning hyperparameters: {e}")
            raise

    def load_data(self, filepath: str) -> Tuple[pd.DataFrame, pd.Series]:
        try:
            df = pd.read_csv(filepath)
            logging.info(f"Initial columns: {df.columns.tolist()}")
            logging.info(f"Initial dtypes: {df.dtypes}")

            columns_to_drop = ['match_id', 'team1', 'team2', 'bo_type', 'label', 'slug']
            df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
            
            if 'scrape_date' in df.columns:
                df['scrape_date'] = pd.to_datetime(df['scrape_date'])
                df['year'] = df['scrape_date'].dt.year
                df['month'] = df['scrape_date'].dt.month
                df['day'] = df['scrape_date'].dt.day
                df = df.drop(columns=['scrape_date'])
            
            if 'winner' not in df.columns:
                df['winner'] = (df['team1_score'] > df['team2_score']).astype(int)
            
            if len(df) > 20:
                self.holdout_data = df.tail(20)
                df = df.iloc[:-20]
            
            X = df.drop(columns=['winner', 'team1_score', 'team2_score'])
            y = df['winner']
            
            for col in X.columns:
                try:
                    X[col] = pd.to_numeric(X[col])
                except Exception as e:
                    logging.warning(f"Could not convert column {col} to numeric: {e}")
                    X = X.drop(columns=[col])
            
            X = X.fillna(0)
            
            logging.info(f"Final columns: {X.columns.tolist()}")
            logging.info(f"Final dtypes: {X.dtypes}")
            
            return X, y
            
        except Exception as e:
            logging.error(f"Error loading data: {e}")
            raise

    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        try:
            X_train = X_train.select_dtypes(include=['number'])
            self.ensemble.fit(X_train, y_train)
        except Exception as e:
            logging.error(f"Error training model: {e}")
            raise

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> None:
        try:
            y_pred = self.model.predict(X_test)
            y_prob = self.model.predict_proba(X_test)[:, 1]
            
            accuracy = accuracy_score(y_test, y_pred)
            brier = brier_score_loss(y_test, y_prob)
            logloss = log_loss(y_test, y_prob)
            roc_auc = roc_auc_score(y_test, y_prob)
            
            logging.info(f"Test set metrics:")
            logging.info(f"  Accuracy: {accuracy:.4f}")
            logging.info(f"  Brier score: {brier:.4f}")
            logging.info(f"  Log loss: {logloss:.4f}")
            logging.info(f"  ROC AUC: {roc_auc:.4f}")
            
            sample = pd.DataFrame({
                "winner_true": y_test.values,
                "winner_pred": y_pred,
                "prob_team1_win": y_prob
            })
            logging.info(f"\nSample predictions:\n{sample.head()}")
            
        except Exception as e:
            logging.error(f"Error evaluating model: {e}")
            raise

    def save_model(self, feature_columns_path: str = "./model/feature_columns.joblib",
                  model_path: str = "./model/model.joblib") -> None:
        try:
            Path(model_path).parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(self.model, model_path, compress=3)
            joblib.dump(self.feature_columns, feature_columns_path)
            logging.info(f"Model saved to {model_path}")
            logging.info(f"Feature columns saved to {feature_columns_path}")
        except Exception as e:
            logging.error(f"Error saving model: {e}")
            raise

    def run(self) -> None:
        try:
            X, y = self.load_data(self.data_path)
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            best_params = self.tune_hyperparameters(X_train, y_train)
            
            self.train(X_train, y_train)
            
            self.evaluate(X_test, y_test)
            
            self.save_model()
            
        except Exception as e:
            logging.error(f"Error in training pipeline: {e}")
            raise
