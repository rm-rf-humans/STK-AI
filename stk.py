import numpy as np
import cv2
import time
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from pynput import keyboard
import os
from torch.utils.data import Dataset, DataLoader
import subprocess
import tempfile
from PIL import Image
import sys
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import seaborn as sns

class ScreenCapture:
    def __init__(self):
        # Create temporary directory for screenshots
        self.temp_dir = tempfile.mkdtemp()
        self.temp_file = os.path.join(self.temp_dir, "screenshot.png")
        
        # Get the game window coordinates using slurp
        try:
            print("Please select the SuperTuxKart window with your mouse...")
            result = subprocess.run(['slurp'], capture_output=True, text=True)
            if result.returncode != 0:
                print("Failed to select window area. Make sure slurp is installed.")
                sys.exit(1)
            self.geometry = result.stdout.strip()
        except FileNotFoundError:
            print("Error: slurp not found. Please install with: sudo pacman -S slurp")
            sys.exit(1)

    def capture_screen(self):
        try:
            # Capture screenshot using grim
            subprocess.run(['grim', '-g', self.geometry, self.temp_file], check=True)
            
            # Read and process the screenshot
            img = cv2.imread(self.temp_file)
            if img is None:
                raise Exception("Failed to read screenshot")
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (84, 84))  # Resize to standard ML input size
            return img
            
        except FileNotFoundError:
            print("Error: grim not found. Please install with: sudo pacman -S grim")
            sys.exit(1)
        except subprocess.CalledProcessError as e:
            print(f"Screenshot failed: {e}")
            return None
    
    def __del__(self):
        # Cleanup temporary files
        try:
            os.remove(self.temp_file)
            os.rmdir(self.temp_dir)
        except:
            pass

class EnhancedSTKModel(nn.Module):
    def __init__(self, num_actions=6):  # Left, Right, Accelerate, Brake, Nitro, Drift
        super(EnhancedSTKModel, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Calculate the output size of conv layers
        self.conv_output_size = 7 * 7 * 128
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.conv_output_size, 512)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_actions)
        
        # Steering head
        self.steering_head = nn.Linear(256, 1)
        
        # Activation functions
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        # Convolutional layers with batch norm
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        
        # Flatten the output
        x = x.reshape(x.size(0), -1)
        
        # Fully connected layers
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        
        # Split into actions and steering
        actions = self.fc3(x)
        steering = self.tanh(self.steering_head(x))
        
        return actions, steering

# The rest of the code remains the same as in the previous version...

class KeyboardMonitor:
    def __init__(self):
        self.current_keys = set()
        self.listener = keyboard.Listener(
            on_press=self.on_press,
            on_release=self.on_release)
        self.listener.start()

    def on_press(self, key):
        try:
            self.current_keys.add(key.char)
        except AttributeError:
            self.current_keys.add(str(key))

    def on_release(self, key):
        try:
            self.current_keys.discard(key.char)
        except AttributeError:
            self.current_keys.discard(str(key))

    def get_action(self):
        action = np.zeros(6)  # [left, right, up, down, nitro, drift]
        if "'a'" in self.current_keys or 'Key.left' in self.current_keys: action[0] = 1
        if "'d'" in self.current_keys or 'Key.right' in self.current_keys: action[1] = 1
        if "'w'" in self.current_keys or 'Key.up' in self.current_keys: action[2] = 1
        if "'s'" in self.current_keys or 'Key.down' in self.current_keys: action[3] = 1
        if "'n'" in self.current_keys: action[4] = 1  # Nitro
        if "'space'" in self.current_keys: action[5] = 1  # Drift
        return action

class VisualTrainer:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.writer = SummaryWriter('runs/stk_training')
        
        plt.ion()
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(15, 5))
        self.loss_history = []
        self.action_history = []
        
    def visualize_training(self, epoch, loss, actions):
        self.loss_history.append(loss)
        self.action_history.append(actions.mean(dim=0).detach().cpu().numpy())
        
        self.ax1.clear()
        self.ax1.plot(self.loss_history)
        self.ax1.set_title('Training Loss')
        self.ax1.set_xlabel('Iteration')
        self.ax1.set_ylabel('Loss')
        
        self.ax2.clear()
        action_data = np.array(self.action_history)
        sns.heatmap(action_data.T, ax=self.ax2, cmap='viridis')
        self.ax2.set_title('Action Distribution Over Time')
        self.ax2.set_xlabel('Iteration')
        self.ax2.set_yticklabels(['Left', 'Right', 'Accel', 'Brake', 'Nitro', 'Drift'])
        
        plt.pause(0.01)
        
        self.writer.add_scalar('Loss/train', loss, epoch)
        for i, action_name in enumerate(['left', 'right', 'accel', 'brake', 'nitro', 'drift']):
            self.writer.add_scalar(f'Actions/{action_name}', actions.mean(dim=0)[i], epoch)

class STKDataset(Dataset):
    def __init__(self, states, actions):
        self.states = states
        self.actions = actions

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx]

def collect_training_data(screen_capture, keyboard_monitor, num_samples=1000):
    states = []
    actions = []
    print("Collecting training data... Press 'q' to stop")
    
    while len(states) < num_samples:
        state = screen_capture.capture_screen()
        if state is None:
            continue
            
        action = keyboard_monitor.get_action()
        
        states.append(state)
        actions.append(action)
        
        if 'q' in keyboard_monitor.current_keys:
            break
        
        time.sleep(0.1)  # Collect data at 10Hz
    
    return np.array(states), np.array(actions)

def train_model(model, states, actions, device, epochs=10, batch_size=32):
    dataset = STKDataset(
        torch.FloatTensor(states.transpose(0, 3, 1, 2) / 255.0),
        torch.FloatTensor(actions)
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)
    
    visualizer = VisualTrainer(model, device)
    
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (batch_states, batch_actions) in enumerate(dataloader):
            batch_states = batch_states.to(device)
            batch_actions = batch_actions.to(device)
            
            optimizer.zero_grad()
            action_preds, steering = model(batch_states)
            
            action_loss = criterion(action_preds, batch_actions)
            loss = action_loss
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                visualizer.visualize_training(
                    epoch * len(dataloader) + batch_idx,
                    loss.item(),
                    torch.sigmoid(action_preds)
                )
        
        avg_loss = total_loss / len(dataloader)
        scheduler.step(avg_loss)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
    
    plt.ioff()
    visualizer.writer.close()

def main():
    # Check for required tools
    for tool in ['grim', 'slurp']:
        if subprocess.run(['which', tool], capture_output=True).returncode != 0:
            print(f"Error: {tool} not found. Please install with: sudo pacman -S {tool}")
            sys.exit(1)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    screen_capture = ScreenCapture()
    keyboard_monitor = KeyboardMonitor()
    model = EnhancedSTKModel().to(device)
    
    # Create save directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    print("Starting data collection... Use these controls:")
    print("W/↑: Accelerate")
    print("S/↓: Brake")
    print("A/←: Steer Left")
    print("D/→: Steer Right")
    print("N: Nitro")
    print("Space: Drift")
    print("Q: Quit collection")
    
    states, actions = collect_training_data(screen_capture, keyboard_monitor)
    print(f"Collected {len(states)} samples")
    
    print("Starting training with visualization...")
    train_model(model, states, actions, device)
    
    torch.save(model.state_dict(), "models/enhanced_stk_model.pth")
    print("Model saved to models/enhanced_stk_model.pth")
    
    print("Starting inference mode... Press 'q' to quit")
    model.eval()
    controller = keyboard.Controller()
    
    while True:
        state = screen_capture.capture_screen()
        if state is None:
            continue
            
        state_tensor = torch.FloatTensor(state.transpose(2, 0, 1) / 255.0).unsqueeze(0).to(device)
        
        with torch.no_grad():
            action_probs, steering = model(state_tensor)
            action_probs = torch.sigmoid(action_probs).cpu().numpy()[0]
            steering_value = steering.cpu().numpy()[0][0]
        
        # Convert predictions to keyboard commands with smoothing
        keys = [keyboard.Key.left, keyboard.Key.right, keyboard.Key.up, keyboard.Key.down, 
                keyboard.KeyCode.from_char('n'), keyboard.Key.space]
                
        # Apply steering based on continuous prediction
        if steering_value < -0.3:
            controller.press(keys[0])  # Left
            controller.release(keys[1])  # Right
        elif steering_value > 0.3:
            controller.press(keys[1])  # Right
            controller.release(keys[0])  # Left
        else:
            controller.release(keys[0])
            controller.release(keys[1])
        
        # Apply other actions based on thresholds
        for key, prob in zip(keys[2:], action_probs[2:]):
            if prob > 0.5:
                controller.press(key)
            else:
                controller.release(key)
        
        if 'q' in keyboard_monitor.current_keys:
            break
        
        time.sleep(0.05)  # Increased response rate

if __name__ == "__main__":
    main()
