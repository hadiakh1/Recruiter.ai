"""
Base Agent Class
Foundation for all agents in the multi-agent system
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from threading import Lock

class SharedMemory:
    """Shared memory object for inter-agent communication"""
    def __init__(self):
        self.data = {}
        self.lock = Lock()
    
    def set(self, key: str, value: Any):
        """Set a value in shared memory"""
        with self.lock:
            self.data[key] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from shared memory"""
        with self.lock:
            return self.data.get(key, default)
    
    def update(self, updates: Dict):
        """Update multiple values at once"""
        with self.lock:
            self.data.update(updates)
    
    def clear(self):
        """Clear all data"""
        with self.lock:
            self.data.clear()

class BaseAgent(ABC):
    """Base class for all agents"""
    
    def __init__(self, agent_id: str, shared_memory: Optional[SharedMemory] = None):
        self.agent_id = agent_id
        self.shared_memory = shared_memory or SharedMemory()
        self.status = "idle"
        self.results = {}
    
    @abstractmethod
    def process(self, input_data: Dict) -> Dict:
        """Process input data and return results"""
        pass
    
    def publish(self, topic: str, data: Any):
        """Publish data to shared memory (publish/subscribe pattern)"""
        self.shared_memory.set(f"{topic}_{self.agent_id}", data)
    
    def subscribe(self, topic: str, agent_id: Optional[str] = None) -> Any:
        """Subscribe to data from shared memory"""
        key = f"{topic}_{agent_id}" if agent_id else topic
        return self.shared_memory.get(key)
    
    def set_status(self, status: str):
        """Set agent status"""
        self.status = status
    
    def get_status(self) -> str:
        """Get agent status"""
        return self.status





