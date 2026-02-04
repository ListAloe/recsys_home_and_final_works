"""Нейросетевая модель Two-Tower для рекомендаций."""

from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader as TorchDataLoader


class InteractionDataset(Dataset):
    """Датасет PyTorch для обучения нейросетевой модели рекомендаций.
    
    Преобразует таблицу рейтингов в примеры для обучения.
    """
    
    def __init__(self, user_ids: np.ndarray, item_ids: np.ndarray,
                 ratings: np.ndarray, negative_samples: int = 1):
        """
        Инициализация датасета.
        
        Args:
            user_ids: Массив ID пользователей
            item_ids: Массив ID книг (позитивные примеры)
            ratings: Массив оценок для каждой пары (user, item)
            negative_samples: Количество негативных примеров на один позитивный
        """
        self.user_ids = user_ids
        self.item_ids = item_ids
        self.ratings = ratings
        self.negative_samples = negative_samples
    
    def __len__(self) -> int:
        return len(self.user_ids)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Получение одного примера из датасета.
        
        Returns:
            Кортеж (user_id, item_id, rating) как torch тензоры с корректными типами данных
        """
        return (
            torch.tensor(self.user_ids[idx], dtype=torch.long),
            torch.tensor(self.item_ids[idx], dtype=torch.long),
            torch.tensor(self.ratings[idx], dtype=torch.float32)
        )


class TwoTowerModel(nn.Module):
    """Two-Tower архитектура для нейросетевой рекомендации.
    
    Содержит две независимые башни:
    - User tower: преобразует embeddings пользователя в латентное пространство
    - Item tower: преобразует embeddings товара в то же латентное пространство
    
    Предсказание рейтинга: score = dot_product(user_vector, item_vector)
    """
    
    def __init__(self, n_users: int, n_items: int, embedding_dim: int = 64,
                 hidden_dims: List[int] = None, n_user_features: int = 0,
                 n_item_features: int = 0):
        """
        Инициализация Two-Tower модели.
        
        Args:
            n_users: Количество уникальных пользователей
            n_items: Количество уникальных книг
            embedding_dim: Размерность embedding слоев и выходных векторов
            hidden_dims: Размеры скрытых слоев в MLP (если None, используется [128, 64])
            n_user_features: Количество дополнительных признаков пользователя
            n_item_features: Количество дополнительных признаков товара
        """
        super().__init__()
        
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.n_user_features = n_user_features
        self.n_item_features = n_item_features
        
        if hidden_dims is None:
            hidden_dims = [128, 64]
        
        # Башня пользователя
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.user_layers = nn.ModuleList()
        
        user_input_dim = embedding_dim + n_user_features
        prev_dim = user_input_dim
        for hidden_dim in hidden_dims:
            self.user_layers.append(nn.Linear(prev_dim, hidden_dim))
            self.user_layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        self.user_output = nn.Linear(prev_dim, embedding_dim)
        
        # Башня товаров
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        self.item_layers = nn.ModuleList()
        
        item_input_dim = embedding_dim + n_item_features
        prev_dim = item_input_dim
        for hidden_dim in hidden_dims:
            self.item_layers.append(nn.Linear(prev_dim, hidden_dim))
            self.item_layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        self.item_output = nn.Linear(prev_dim, embedding_dim)
        
        # Инициализация весов
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Инициализировать веса сети."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.01)
    
    def get_user_embedding(self, user_ids: torch.Tensor,
                          user_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Получить embedding пользователя."""
        x = self.user_embedding(user_ids)
        
        # Конкатенировать с дополнительными признаками
        if user_features is not None and self.n_user_features > 0:
            x = torch.cat([x, user_features], dim=1)
        
        for layer in self.user_layers:
            x = layer(x)
        
        x = self.user_output(x)
        x = torch.nn.functional.normalize(x, p=2, dim=1)
        
        return x
    
    def get_item_embedding(self, item_ids: torch.Tensor,
                          item_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Получить embedding книги."""
        x = self.item_embedding(item_ids)
        
        # Конкатенировать с дополнительными признаками
        if item_features is not None and self.n_item_features > 0:
            x = torch.cat([x, item_features], dim=1)
        
        for layer in self.item_layers:
            x = layer(x)
        
        x = self.item_output(x)
        x = torch.nn.functional.normalize(x, p=2, dim=1)
        
        return x
    
    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor,
                user_features: Optional[torch.Tensor] = None,
                item_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            user_ids: tensor shape (batch_size,)
            item_ids: tensor shape (batch_size,)
            user_features: опциональный tensor shape (batch_size, n_user_features)
            item_features: опциональный tensor shape (batch_size, n_item_features)
            
        Returns:
            Predictions shape (batch_size,)
        """
        user_emb = self.get_user_embedding(user_ids, user_features)
        item_emb = self.get_item_embedding(item_ids, item_features)
        
        # Скалярное произведение (подобие)
        predictions = torch.sum(user_emb * item_emb, dim=1)
        
        return predictions


class NeuralRecommender:
    """Нейросетевая рекомендательная система с поддержкой дополнительных признаков."""
    
    def __init__(self, n_users: int, n_items: int, embedding_dim: int = 64,
                 hidden_dims: List[int] = None, device: str = 'cpu',
                 learning_rate: float = 0.001, n_user_features: int = 0,
                 n_item_features: int = 0, random_state: int = 42):
        """
        Инициализация.
        
        Args:
            n_users: количество пользователей
            n_items: количество книг
            embedding_dim: размерность embedding
            hidden_dims: размеры скрытых слоёв
            device: 'cpu' или 'cuda'
            learning_rate: learning rate оптимайзера
            n_user_features: количество дополнительных признаков пользователя
            n_item_features: количество дополнительных признаков книги
            random_state: random seed для воспроизводимости
        """
        self.device = torch.device(device)
        self.n_users = n_users
        self.n_items = n_items
        self.learning_rate = learning_rate
        self.n_user_features = n_user_features
        self.n_item_features = n_item_features
        self.random_state = random_state
        
        # Установка seed
        torch.manual_seed(random_state)
        np.random.seed(random_state)
        
        self.model = TwoTowerModel(
            n_users, n_items,
            embedding_dim=embedding_dim,
            hidden_dims=hidden_dims,
            n_user_features=0,
            n_item_features=0
        ).float().to(self.device)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        
        self.user_id_to_idx = None
        self.item_id_to_idx = None
        self.idx_to_item_id = None
        self.user_features_dict = None
        self.item_features_dict = None
        self.training_history = []
    
    def prepare_data(self, train_df: pd.DataFrame, user_ids: List[int],
                    item_ids: List[int], user_features: pd.DataFrame = None,
                    item_features: pd.DataFrame = None) -> None:
        """
        Подготовить маппинги ID и признаки.
        
        Args:
            train_df: обучающие данные
            user_ids: уникальные user_ids
            item_ids: уникальные item_ids
            user_features: DataFrame с признаками пользователей (индекс = user_id)
            item_features: DataFrame с признаками книг (индекс = book_id)
        """
        self.user_id_to_idx = {uid: idx for idx, uid in enumerate(user_ids)}
        self.item_id_to_idx = {iid: idx for idx, iid in enumerate(item_ids)}
        self.idx_to_item_id = {idx: iid for iid, idx in self.item_id_to_idx.items()}
        
        if user_features is not None:
            self.user_features_dict = user_features.to_dict('index')
        
        if item_features is not None:
            self.item_features_dict = item_features.to_dict('index')
    
    def train_epoch(self, train_df: pd.DataFrame, batch_size: int = 32) -> float:
        """
        Обучить модель на одну эпоху.
        
        Args:
            train_df: обучающие данные
            batch_size: размер батча
            
        Returns:
            Средняя потеря за эпоху
        """
        self.model.train()
        self.model = self.model.float()  # Проверка что модель в float32
        
        # Преображение user_ids и item_ids в индексы
        user_indices = np.array([
            self.user_id_to_idx[uid] for uid in train_df['user_id']
        ], dtype=np.int64)
        item_indices = np.array([
            self.item_id_to_idx[iid] for iid in train_df['book_id']
        ], dtype=np.int64)
        ratings = train_df['rating'].values.astype(np.float32)
        
        # Нормализация рейтингов в [0, 1]
        ratings_normalized = (ratings - 1.0) / 4.0
        
        dataset = InteractionDataset(user_indices, item_indices, ratings_normalized)
        dataloader = TorchDataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        total_loss = 0.0
        n_batches = 0
        
        for batch_user_ids, batch_item_ids, batch_ratings in dataloader:
            batch_user_ids = batch_user_ids.to(self.device)
            batch_item_ids = batch_item_ids.to(self.device)
            batch_ratings = batch_ratings.to(self.device).float().unsqueeze(1)
            
            # Получение признаков, если доступны
            batch_user_features = None
            batch_item_features = None
            
            if self.user_features_dict is not None and self.n_user_features > 0:
                batch_user_features = []
                for idx in batch_user_ids.cpu().numpy():
                    # Поиск user_id по индексу
                    for uid, uidx in self.user_id_to_idx.items():
                        if uidx == idx:
                            if uid in self.user_features_dict:
                                feats = list(self.user_features_dict[uid].values())
                                batch_user_features.append(feats)
                            break
                
                if batch_user_features and len(batch_user_features) == len(batch_user_ids):
                    batch_user_features = torch.tensor(batch_user_features,
                                                       dtype=torch.float32,
                                                       device=self.device)
                else:
                    batch_user_features = None
            
            if self.item_features_dict is not None and self.n_item_features > 0:
                batch_item_features = []
                for idx in batch_item_ids.cpu().numpy():
                    # Поиск item_id по индексу
                    if idx in self.idx_to_item_id:
                        iid = self.idx_to_item_id[idx]
                        if iid in self.item_features_dict:
                            feats = list(self.item_features_dict[iid].values())
                            batch_item_features.append(feats)
                
                if batch_item_features and len(batch_item_features) == len(batch_item_ids):
                    batch_item_features = torch.tensor(batch_item_features,
                                                       dtype=torch.float32,
                                                       device=self.device)
                else:
                    batch_item_features = None
            
            # Прямой проход
            predictions = self.model(batch_user_ids, batch_item_ids,
                                    batch_user_features, batch_item_features).unsqueeze(1)
            
            # Проверка что предсказания и рейтинги float32
            predictions = predictions.float()
            batch_ratings = batch_ratings.float()
            
            # Вычисление потери
            loss = self.criterion(predictions, batch_ratings)
            
            # Обратный проход
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        avg_loss = total_loss / n_batches if n_batches > 0 else 0.0
        self.training_history.append(avg_loss)
        
        return avg_loss
    
    def fit(self, train_df: pd.DataFrame, user_ids: List[int], item_ids: List[int],
            epochs: int = 10, batch_size: int = 32, verbose: bool = True,
            user_features: pd.DataFrame = None,
            item_features: pd.DataFrame = None) -> 'NeuralRecommender':
        """
        Обучить модель.
        
        Args:
            train_df: обучающие данные
            user_ids: уникальные user_ids
            item_ids: уникальные item_ids
            epochs: количество эпох
            batch_size: размер батча
            verbose: выводить ли прогресс
            user_features: DataFrame с признаками пользователей
            item_features: DataFrame с признаками книг
            
        Returns:
            self
        """
        self.prepare_data(train_df, user_ids, item_ids, user_features, item_features)
        
        print(f"Обучение Two-Tower модели в течение {epochs} эпох...")
        
        for epoch in range(epochs):
            avg_loss = self.train_epoch(train_df, batch_size=batch_size)
            
            if verbose and (epoch + 1) % max(1, epochs // 5) == 0:
                print(f"  Эпоха {epoch + 1}/{epochs}: Loss = {avg_loss:.4f}")
        
        print("Обучение завершено")
        
        return self
    
    def predict_rating(self, user_id: int, item_id: int) -> float:
        """
        Предсказать рейтинг пользователя для книги.
        
        Args:
            user_id: ID пользователя
            item_id: ID книги
            
        Returns:
            Предсказанный рейтинг (1-5)
        """
        if user_id not in self.user_id_to_idx or item_id not in self.item_id_to_idx:
            return 3.0  # Дефолтный рейтинг
        
        self.model.eval()
        
        with torch.no_grad():
            user_idx = torch.tensor([self.user_id_to_idx[user_id]], device=self.device)
            item_idx = torch.tensor([self.item_id_to_idx[item_id]], device=self.device)
            
            prediction = self.model(user_idx, item_idx).item()
        
        # Денормализация рейтинга обратно в [1, 5]
        rating = prediction * 4.0 + 1.0
        
        return float(np.clip(rating, 1.0, 5.0))
    
    def get_recommendations(self, user_id: int, top_n: int = 10,
                           exclude_seen: bool = True,
                           train_df: pd.DataFrame = None) -> List[int]:
        """
        Получить рекомендации для пользователя (батч-операция).
        
        Args:
            user_id: ID пользователя
            top_n: количество рекомендаций
            exclude_seen: исключать ли оценённые книги
            train_df: обучающие данные для определения seen items
            
        Returns:
            Список ID книг
        """
        if user_id not in self.user_id_to_idx:
            return []
        
        self.model.eval()
        
        with torch.no_grad():
            user_idx = torch.tensor([self.user_id_to_idx[user_id]], device=self.device)
            user_emb = self.model.get_user_embedding(user_idx)  # (1, embedding_dim)
            
            # Батч-операция: вычисление всех embedding-ов товаров сразу
            all_item_indices = torch.arange(self.n_items, device=self.device)
            all_item_embs = self.model.get_item_embedding(all_item_indices)  # (n_items, embedding_dim)
            
            # Вычисление скалярного произведения между пользователем и всеми товарами
            predictions = torch.mm(user_emb, all_item_embs.t()).squeeze(0).cpu().numpy()  # (n_items,)
        
        # Исключение оцененных книг
        if exclude_seen and train_df is not None:
            seen_books = set(train_df[train_df['user_id'] == user_id]['book_id'])
            for item_id in seen_books:
                if item_id in self.item_id_to_idx:
                    idx = self.item_id_to_idx[item_id]
                    predictions[idx] = -np.inf
        
        # Получение топ-рекомендаций
        top_indices = np.argsort(predictions)[::-1][:top_n]
        
        recommendations = [
            self.idx_to_item_id[idx]
            for idx in top_indices
            if predictions[idx] > -np.inf
        ]
        
        return recommendations
    
    def get_all_recommendations(self, test_df: pd.DataFrame, top_n: int = 10,
                               train_df: pd.DataFrame = None, verbose: bool = False) -> Dict[int, List[int]]:
        """
        Получить рекомендации для всех пользователей в тесте.
        
        Args:
            test_df: тестовые данные
            top_n: количество рекомендаций на пользователя
            train_df: обучающие данные
            verbose: выводить прогресс
            
        Returns:
            Словарь {user_id: [book_ids]}
        """
        recommendations = {}
        unique_users = test_df['user_id'].unique()
        n_users = len(unique_users)
        
        for idx, user_id in enumerate(unique_users):
            if verbose and (idx + 1) % max(1, n_users // 10) == 0:
                print(f"  Обработано {idx + 1}/{n_users} пользователей")
            
            recs = self.get_recommendations(
                user_id, top_n=top_n,
                exclude_seen=True, train_df=train_df
            )
            recommendations[user_id] = recs
        
        if verbose:
            print(f"  Обработано {n_users}/{n_users} пользователей")
        
        return recommendations
