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

# Define the CNN model
class STKModel(nn.Module):
    def __init__(self, num_actions=4):  # Left, Right, Accelerate, Brake
        super(STKModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(7 * 7 * 64, 512)
        self.fc2 = nn.Linear(512, num_actions)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)

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

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        states, actions, rewards, next_states, dones = zip(*np.random.choice(self.buffer, batch_size))
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))

    def __len__(self):
        return len(self.buffer)

class STKDataset(Dataset):
    def __init__(self, states, actions):
        self.states = states
        self.actions = actions

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx]

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
        action = np.zeros(4)  # [left, right, up, down]
        if "'a'" in self.current_keys or 'Key.left' in self.current_keys: action[0] = 1
        if "'d'" in self.current_keys or 'Key.right' in self.current_keys: action[1] = 1
        if "'w'" in self.current_keys or 'Key.up' in self.current_keys: action[2] = 1
        if "'s'" in self.current_keys or 'Key.down' in self.current_keys: action[3] = 1
        return action

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
    
    for epoch in range(epochs):
        total_loss = 0
        for batch_states, batch_actions in dataloader:
            batch_states = batch_states.to(device)
            batch_actions = batch_actions.to(device)
            
            optimizer.zero_grad()
            predictions = model(batch_states)
            loss = criterion(predictions, batch_actions)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(dataloader):.4f}")

def main():
    # Check for required tools
    for tool in ['grim', 'slurp']:
        if subprocess.run(['which', tool], capture_output=True).returncode != 0:
            print(f"Error: {tool} not found. Please install with: sudo pacman -S {tool}")
            sys.exit(1)
    
    # Set up device (CPU in this case)
    device = torch.device("cpu")
    
    # Initialize components
    screen_capture = ScreenCapture()
    keyboard_monitor = KeyboardMonitor()
    model = STKModel().to(device)
    
    # Create save directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # Collect training data
    print("Starting data collection...")
    states, actions = collect_training_data(screen_capture, keyboard_monitor)
    print(f"Collected {len(states)} samples")
    
    # Train the model
    print("Starting training...")
    train_model(model, states, actions, device)
    
    # Save the model
    torch.save(model.state_dict(), "models/stk_model.pth")
    print("Model saved to models/stk_model.pth")
    
    # Run inference
    print("Starting inference mode... Press 'q' to quit")
    model.eval()
    
    controller = keyboard.Controller()
    
    while True:
        state = screen_capture.capture_screen()
        if state is None:
            continue
            
        state_tensor = torch.FloatTensor(state.transpose(2, 0, 1) / 255.0).unsqueeze(0).to(device)
        
        with torch.no_grad():
            action_probs = model(state_tensor).cpu().numpy()[0]
        
        # Convert predictions to keyboard commands
        keys = [keyboard.Key.left, keyboard.Key.right, keyboard.Key.up, keyboard.Key.down]
        for key, prob in zip(keys, action_probs):
            if prob > 0.5:
                controller.press(key)
            else:
                controller.release(key)
        
        if 'q' in keyboard_monitor.current_keys:
            break
        
        time.sleep(0.1)

if __name__ == "__main__":
    main()
