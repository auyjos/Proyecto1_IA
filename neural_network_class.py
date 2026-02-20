import numpy as np

class NeuralNetwork:
    def __init__(self, layer_sizes):
        """
        layer_sizes: lista con el número de neuronas por capa, ej: [3, 32, 16, 5]
        Inicializa pesos y sesgos para cada capa.
        """
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes) - 1  # No cuenta la entrada
        self.weights = []  # Lista de matrices de pesos
        self.biases = []   # Lista de vectores de sesgos

        for i in range(self.num_layers):
            input_size = layer_sizes[i]
            output_size = layer_sizes[i+1]
            # Pesos: matriz (input_size, output_size)
            W = np.random.randn(input_size, output_size) * 0.01
            # Sesgos: vector (output_size,)
            b = np.zeros(output_size)
            self.weights.append(W)
            self.biases.append(b)

    def relu(self, z):
        """
        Función de activación ReLU: max(0, z)
        Entrada: matriz z (cualquier forma)
        Salida: matriz con valores negativos → 0, positivos → igual
        """
        return np.maximum(0, z)

    def softmax(self, z):
        """
        Función softmax: convierte logits en probabilidades que suman 1
        Entrada: matriz z con shape (N, num_clases)
        Salida: matriz con probabilidades [0, 1] que suma 1 por fila
        """
        # Restar max por estabilidad numérica
        z_shifted = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z_shifted)
        softmax_z = exp_z / np.sum(exp_z, axis=1, keepdims=True)
        return softmax_z
    
    def forward(self, X):
        """
        Propagación hacia adelante: calcula la salida de la red para una entrada X
        Retorna AMBAS: output y cache (valores intermedios para backprop)
        Entrada: X con shape (N, num_entradas)
        Salida: (probabilidades con shape (N, num_clases), cache de valores intermedios)
        """
        cache = {}  # Guardar valores intermedios para usar en backward
        A = X
        
        for i in range(self.num_layers):
            cache[f'A{i}'] = A  # Guardar activación anterior
            W = self.weights[i]
            b = self.biases[i]
            Z = A @ W + b  # Producto matricial + sesgo
            cache[f'Z{i}'] = Z  # Guardar Z antes de activación
            
            if i < self.num_layers - 1:
                A = self.relu(Z)  # ReLU para capas ocultas
            else:
                A = self.softmax(Z)  # Softmax para capa de salida
        
        cache[f'A{self.num_layers}'] = A  # Guardar output final
        return A, cache
    
    def cross_entropy_loss(self, Y_pred, Y_true):
        """
        Calcula la pérdida de entropía cruzada entre las predicciones y las etiquetas verdaderas
        Entrada: 
            Y_pred: matriz de probabilidades (N, num_clases)
            Y_true: matriz one-hot de etiquetas verdaderas (N, num_clases)
        Salida: pérdida escalar
        """
        N = Y_pred.shape[0]
        # Evitar log(0) sumando una pequeña constante
        epsilon = 1e-15
        Y_pred_clipped = np.clip(Y_pred, epsilon, 1 - epsilon)
        loss = -np.sum(Y_true * np.log(Y_pred_clipped)) / N
        return loss
    
    def backward(self, X, Y_true, Y_pred, cache):
        """
        Propagación hacia atrás: calcula los gradientes de pesos y sesgos
        Entrada:
            X: matriz de entrada (N, num_entradas)
            Y_true: matriz one-hot de etiquetas verdaderas (N, num_clases)
            Y_pred: matriz de probabilidades predichas (N, num_clases)
            cache: diccionario con valores intermedios guardados en forward
        Salida:
            grad_weights: lista de matrices de gradientes para pesos
            grad_biases: lista de vectores de gradientes para sesgos
        """
        N = X.shape[0]
        grad_weights = [None] * self.num_layers
        grad_biases = [None] * self.num_layers
        
        # Gradiente inicial: salida - etiqueta (derivada de cross_entropy + softmax)
        dZ = Y_pred - Y_true  # shape: (N, num_clases)
        
        # Iterar hacia atrás a través de todas las capas
        for i in reversed(range(self.num_layers)):
            A_prev = cache[f'A{i}']  # Activación anterior
            
            # Calcular gradientes para W y b
            dW = A_prev.T @ dZ / N
            db = np.sum(dZ, axis=0) / N
            
            grad_weights[i] = dW
            grad_biases[i] = db
            
            # Si no es la primera capa, propagar el gradiente hacia atrás
            if i > 0:
                W = self.weights[i]
                dA_prev = dZ @ W.T  # Gradiente de la activación anterior
                Z_prev = cache[f'Z{i-1}']  # Z de la capa anterior
                dZ = dA_prev * (Z_prev > 0)  # ReLU backward (derivada)
        
        return grad_weights, grad_biases
    
    def update_weights(self, grad_weights, grad_biases, learning_rate):
        """
        Actualiza los pesos y sesgos usando Descenso de Gradiente Estocástico (SGD)
        Entrada:
            grad_weights: lista de gradientes para cada peso
            grad_biases: lista de gradientes para cada sesgo
            learning_rate: tasa de aprendizaje (ej: 0.01)
        """
        for i in range(self.num_layers):
            self.weights[i] -= learning_rate * grad_weights[i]
            self.biases[i] -= learning_rate * grad_biases[i]
    
    def train(self, X_train, Y_train, epochs=50, learning_rate=0.01, batch_size=32):
        """
        Entrena la red neuronal usando SGD + Backpropagation
        Entrada:
            X_train: datos de entrenamiento (N, num_entradas)
            Y_train: etiquetas de entrenamiento (N, num_clases) - one-hot
            epochs: número de iteraciones sobre el dataset completo
            learning_rate: tasa de aprendizaje (ej: 0.01)
            batch_size: tamaño de cada mini-batch (ej: 32)
        """
        num_samples = X_train.shape[0]
        losses = []
        
        for epoch in range(epochs):
            epoch_loss = 0
            num_batches = 0
            
            # Crear índices aleatorios para mezclar datos
            indices = np.random.permutation(num_samples)
            X_shuffled = X_train[indices]
            Y_shuffled = Y_train[indices]
            
            # Iterar sobre mini-batches
            for start_idx in range(0, num_samples, batch_size):
                end_idx = min(start_idx + batch_size, num_samples)
                X_batch = X_shuffled[start_idx:end_idx]
                Y_batch = Y_shuffled[start_idx:end_idx]
                
                # Forward pass
                Y_pred, cache = self.forward(X_batch)
                
                # Calcular loss
                loss = self.cross_entropy_loss(Y_pred, Y_batch)
                epoch_loss += loss
                num_batches += 1
                
                # Backward pass
                grad_weights, grad_biases = self.backward(X_batch, Y_batch, Y_pred, cache)
                
                # Actualizar pesos
                self.update_weights(grad_weights, grad_biases, learning_rate)
            
            # Promediar loss del epoch
            avg_loss = epoch_loss / num_batches
            losses.append(avg_loss)
            
            # Imprimir cada 10 epochs
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}")
        
        print(f"Entrenamiento completado. Loss final: {losses[-1]:.4f}")
        return losses
    
    def predict(self, X):
        """
        Predice la clase para cada muestra en X
        Entrada: X con shape (N, num_entradas)
        Salida: array de predicciones con shape (N,) - índice de clase (0 a num_clases-1)
        """
        Y_pred, _ = self.forward(X)  # Obtener probabilidades
        predictions = np.argmax(Y_pred, axis=1)  # Índice de la probabilidad máxima
        return predictions
    
    def accuracy(self, y_pred, y_true):
        """
        Calcula la precisión (accuracy) entre predicciones y etiquetas verdaderas
        Entrada:
            y_pred: array de predicciones (N,) - índices de clases
            y_true: array de etiquetas verdaderas (N,) - índices de clases
        Salida: porcentaje de precisión (0-100)
        """
        correct = np.sum(y_pred == y_true)
        total = len(y_true)
        accuracy_percent = (correct / total) * 100
        return accuracy_percent
    

