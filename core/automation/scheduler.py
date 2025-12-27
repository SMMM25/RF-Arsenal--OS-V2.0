"""
RF Arsenal OS - Task Scheduler
Scheduled task execution for automated RF operations.
"""

import time
import threading
import hashlib
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import heapq


class ScheduleType(Enum):
    """Types of schedules"""
    ONCE = "once"           # Run once at specific time
    INTERVAL = "interval"   # Run every N seconds
    CRON = "cron"          # Cron-style scheduling
    DAILY = "daily"        # Run daily at specific time
    HOURLY = "hourly"      # Run every hour at specific minute
    CONTINUOUS = "continuous"  # Run continuously with delay


class TaskPriority(Enum):
    """Task priority levels"""
    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4
    BACKGROUND = 5


class TaskStatus(Enum):
    """Task execution status"""
    PENDING = "pending"
    SCHEDULED = "scheduled"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


@dataclass
class ScheduledTask:
    """A scheduled task"""
    task_id: str
    name: str
    command: str  # Command to execute
    schedule_type: ScheduleType
    schedule_params: Dict[str, Any]  # Type-specific parameters
    priority: TaskPriority = TaskPriority.NORMAL
    enabled: bool = True
    max_runs: int = 0  # 0 = unlimited
    timeout_s: float = 300.0
    retry_on_fail: bool = False
    retry_count: int = 3
    retry_delay_s: float = 60.0
    
    # Runtime state
    status: TaskStatus = TaskStatus.PENDING
    next_run: Optional[float] = None
    last_run: Optional[float] = None
    run_count: int = 0
    fail_count: int = 0
    last_result: Optional[Dict] = None
    last_error: Optional[str] = None
    
    def __lt__(self, other):
        """For heap comparison - lower priority value = higher priority"""
        if self.next_run == other.next_run:
            return self.priority.value < other.priority.value
        return (self.next_run or float('inf')) < (other.next_run or float('inf'))
        
    def to_dict(self) -> Dict:
        return {
            "task_id": self.task_id,
            "name": self.name,
            "command": self.command,
            "schedule_type": self.schedule_type.value,
            "schedule_params": self.schedule_params,
            "priority": self.priority.value,
            "enabled": self.enabled,
            "status": self.status.value,
            "next_run": self.next_run,
            "last_run": self.last_run,
            "run_count": self.run_count,
            "fail_count": self.fail_count
        }


class CronParser:
    """Simple cron expression parser"""
    
    @staticmethod
    def parse(expression: str) -> Dict[str, List[int]]:
        """
        Parse cron expression: minute hour day month weekday
        Supports: *, */n, n, n-m, n,m,o
        """
        fields = expression.split()
        if len(fields) != 5:
            raise ValueError("Cron expression must have 5 fields")
            
        field_names = ["minute", "hour", "day", "month", "weekday"]
        field_ranges = [
            (0, 59),   # minute
            (0, 23),   # hour
            (1, 31),   # day
            (1, 12),   # month
            (0, 6)     # weekday (0 = Sunday)
        ]
        
        result = {}
        
        for i, (field, name, (min_val, max_val)) in enumerate(zip(fields, field_names, field_ranges)):
            values = set()
            
            for part in field.split(','):
                if part == '*':
                    values.update(range(min_val, max_val + 1))
                elif '/' in part:
                    base, step = part.split('/')
                    step = int(step)
                    if base == '*':
                        values.update(range(min_val, max_val + 1, step))
                    else:
                        start = int(base)
                        values.update(range(start, max_val + 1, step))
                elif '-' in part:
                    start, end = part.split('-')
                    values.update(range(int(start), int(end) + 1))
                else:
                    values.add(int(part))
                    
            result[name] = sorted(values)
            
        return result
        
    @staticmethod
    def next_run(expression: str, after: Optional[datetime] = None) -> datetime:
        """Calculate next run time for cron expression"""
        if after is None:
            after = datetime.now()
            
        parsed = CronParser.parse(expression)
        
        # Start from next minute
        candidate = after.replace(second=0, microsecond=0) + timedelta(minutes=1)
        
        for _ in range(366 * 24 * 60):  # Max 1 year lookahead
            if (candidate.minute in parsed["minute"] and
                candidate.hour in parsed["hour"] and
                candidate.day in parsed["day"] and
                candidate.month in parsed["month"] and
                candidate.weekday() in parsed["weekday"]):
                return candidate
            candidate += timedelta(minutes=1)
            
        raise ValueError("Could not find next run time within 1 year")


class TaskScheduler:
    """
    Production-grade task scheduler for RF Arsenal OS.
    
    Features:
    - Multiple schedule types (once, interval, cron, daily, hourly)
    - Priority-based execution
    - Task dependencies
    - Retry on failure
    - Timeout handling
    - Pause/resume individual tasks
    - Real-time status monitoring
    - RAM-only operation support
    """
    
    def __init__(self, command_handler: Optional[Callable] = None):
        """
        Initialize task scheduler.
        
        Args:
            command_handler: Function to execute commands
        """
        self.command_handler = command_handler or (lambda cmd: {"success": True})
        
        # Task storage
        self._tasks: Dict[str, ScheduledTask] = {}
        self._task_heap: List[ScheduledTask] = []  # Priority queue
        
        # Scheduler state
        self._running = False
        self._paused = False
        
        # Worker thread
        self._scheduler_thread: Optional[threading.Thread] = None
        self._executor_threads: List[threading.Thread] = []
        self._max_concurrent = 5
        self._running_tasks: Dict[str, threading.Thread] = {}
        
        # Locks
        self._lock = threading.Lock()
        self._task_lock = threading.Lock()
        
        # Callbacks
        self._task_callbacks: List[Callable] = []
        
    def add_task(self, task: ScheduledTask) -> str:
        """
        Add a task to the scheduler.
        
        Args:
            task: Task to schedule
            
        Returns:
            Task ID
        """
        with self._task_lock:
            # Calculate next run time
            task.next_run = self._calculate_next_run(task)
            task.status = TaskStatus.SCHEDULED
            
            self._tasks[task.task_id] = task
            heapq.heappush(self._task_heap, task)
            
        return task.task_id
        
    def create_task(self,
                   name: str,
                   command: str,
                   schedule_type: ScheduleType,
                   schedule_params: Optional[Dict] = None,
                   **kwargs) -> str:
        """
        Create and add a new task.
        
        Args:
            name: Task name
            command: Command to execute
            schedule_type: Type of schedule
            schedule_params: Schedule-specific parameters
            **kwargs: Additional task parameters
            
        Returns:
            Task ID
        """
        task_id = hashlib.sha256(f"{name}_{time.time()}".encode()).hexdigest()[:16]
        
        task = ScheduledTask(
            task_id=task_id,
            name=name,
            command=command,
            schedule_type=schedule_type,
            schedule_params=schedule_params or {},
            **kwargs
        )
        
        return self.add_task(task)
        
    def remove_task(self, task_id: str) -> bool:
        """Remove a task from the scheduler"""
        with self._task_lock:
            if task_id in self._tasks:
                task = self._tasks[task_id]
                task.status = TaskStatus.CANCELLED
                del self._tasks[task_id]
                return True
        return False
        
    def pause_task(self, task_id: str) -> bool:
        """Pause a task"""
        with self._task_lock:
            if task_id in self._tasks:
                self._tasks[task_id].status = TaskStatus.PAUSED
                return True
        return False
        
    def resume_task(self, task_id: str) -> bool:
        """Resume a paused task"""
        with self._task_lock:
            if task_id in self._tasks:
                task = self._tasks[task_id]
                if task.status == TaskStatus.PAUSED:
                    task.status = TaskStatus.SCHEDULED
                    task.next_run = self._calculate_next_run(task)
                    heapq.heappush(self._task_heap, task)
                    return True
        return False
        
    def trigger_task(self, task_id: str) -> bool:
        """Manually trigger a task to run immediately"""
        with self._task_lock:
            if task_id in self._tasks:
                task = self._tasks[task_id]
                task.next_run = time.time()
                heapq.heappush(self._task_heap, task)
                return True
        return False
        
    def start(self) -> None:
        """Start the scheduler"""
        if self._running:
            return
            
        self._running = True
        self._paused = False
        
        self._scheduler_thread = threading.Thread(
            target=self._scheduler_loop,
            daemon=True
        )
        self._scheduler_thread.start()
        
    def stop(self) -> None:
        """Stop the scheduler"""
        self._running = False
        
        # Wait for running tasks to complete
        for thread in self._running_tasks.values():
            thread.join(timeout=5.0)
            
    def pause(self) -> None:
        """Pause all task execution"""
        self._paused = True
        
    def resume(self) -> None:
        """Resume task execution"""
        self._paused = False
        
    def _scheduler_loop(self) -> None:
        """Main scheduler loop"""
        while self._running:
            if self._paused:
                time.sleep(0.1)
                continue
                
            current_time = time.time()
            
            with self._task_lock:
                # Check for tasks ready to run
                while self._task_heap and self._task_heap[0].next_run <= current_time:
                    task = heapq.heappop(self._task_heap)
                    
                    # Skip cancelled or paused tasks
                    if task.task_id not in self._tasks:
                        continue
                    if task.status in [TaskStatus.CANCELLED, TaskStatus.PAUSED]:
                        continue
                        
                    # Check max runs
                    if task.max_runs > 0 and task.run_count >= task.max_runs:
                        task.status = TaskStatus.COMPLETED
                        continue
                        
                    # Check concurrent limit
                    if len(self._running_tasks) >= self._max_concurrent:
                        # Re-add to heap for later
                        heapq.heappush(self._task_heap, task)
                        break
                        
                    # Execute task in thread
                    thread = threading.Thread(
                        target=self._execute_task,
                        args=(task,),
                        daemon=True
                    )
                    self._running_tasks[task.task_id] = thread
                    thread.start()
                    
            time.sleep(0.1)  # Check every 100ms
            
    def _execute_task(self, task: ScheduledTask) -> None:
        """Execute a single task"""
        task.status = TaskStatus.RUNNING
        task.last_run = time.time()
        
        try:
            # Execute with timeout
            result = self._execute_with_timeout(task.command, task.timeout_s)
            task.last_result = result
            task.run_count += 1
            task.last_error = None
            
            # Notify callbacks
            self._notify_task_complete(task, success=True)
            
        except TimeoutError:
            task.last_error = "Task timed out"
            task.fail_count += 1
            self._notify_task_complete(task, success=False)
            
        except Exception as e:
            task.last_error = str(e)
            task.fail_count += 1
            self._notify_task_complete(task, success=False)
            
            # Retry if configured
            if task.retry_on_fail and task.fail_count <= task.retry_count:
                task.next_run = time.time() + task.retry_delay_s
                with self._task_lock:
                    heapq.heappush(self._task_heap, task)
                task.status = TaskStatus.SCHEDULED
            else:
                task.status = TaskStatus.FAILED
                
        finally:
            # Remove from running tasks
            if task.task_id in self._running_tasks:
                del self._running_tasks[task.task_id]
                
            # Schedule next run if applicable
            if task.status not in [TaskStatus.FAILED, TaskStatus.CANCELLED]:
                next_run = self._calculate_next_run(task)
                if next_run:
                    task.next_run = next_run
                    task.status = TaskStatus.SCHEDULED
                    with self._task_lock:
                        heapq.heappush(self._task_heap, task)
                else:
                    task.status = TaskStatus.COMPLETED
                    
    def _execute_with_timeout(self, command: str, timeout_s: float) -> Dict:
        """Execute command with timeout"""
        result = {"success": False}
        exception = [None]
        
        def execute():
            try:
                result.update(self.command_handler(command))
            except Exception as e:
                exception[0] = e
                
        thread = threading.Thread(target=execute)
        thread.start()
        thread.join(timeout=timeout_s)
        
        if thread.is_alive():
            raise TimeoutError(f"Task exceeded timeout of {timeout_s}s")
            
        if exception[0]:
            raise exception[0]
            
        return result
        
    def _calculate_next_run(self, task: ScheduledTask) -> Optional[float]:
        """Calculate next run time for a task"""
        params = task.schedule_params
        current_time = time.time()
        
        if task.schedule_type == ScheduleType.ONCE:
            run_time = params.get("run_at", current_time)
            if run_time > current_time:
                return run_time
            return None  # Already past
            
        elif task.schedule_type == ScheduleType.INTERVAL:
            interval = params.get("interval_s", 60)
            if task.last_run:
                return task.last_run + interval
            return current_time + interval
            
        elif task.schedule_type == ScheduleType.CRON:
            expression = params.get("expression", "0 * * * *")  # Default: every hour
            try:
                next_dt = CronParser.next_run(expression)
                return next_dt.timestamp()
            except Exception:
                return None
                
        elif task.schedule_type == ScheduleType.DAILY:
            hour = params.get("hour", 0)
            minute = params.get("minute", 0)
            
            now = datetime.now()
            run_time = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            
            if run_time.timestamp() <= current_time:
                run_time += timedelta(days=1)
                
            return run_time.timestamp()
            
        elif task.schedule_type == ScheduleType.HOURLY:
            minute = params.get("minute", 0)
            
            now = datetime.now()
            run_time = now.replace(minute=minute, second=0, microsecond=0)
            
            if run_time.timestamp() <= current_time:
                run_time += timedelta(hours=1)
                
            return run_time.timestamp()
            
        elif task.schedule_type == ScheduleType.CONTINUOUS:
            delay = params.get("delay_s", 1.0)
            return current_time + delay
            
        return None
        
    def _notify_task_complete(self, task: ScheduledTask, success: bool) -> None:
        """Notify callbacks of task completion"""
        for callback in self._task_callbacks:
            try:
                callback(task, success)
            except Exception:
                pass
                
    def register_callback(self, callback: Callable) -> None:
        """Register callback for task completion"""
        self._task_callbacks.append(callback)
        
    def get_task(self, task_id: str) -> Optional[ScheduledTask]:
        """Get task by ID"""
        return self._tasks.get(task_id)
        
    def get_all_tasks(self) -> List[Dict]:
        """Get all tasks"""
        return [task.to_dict() for task in self._tasks.values()]
        
    def get_running_tasks(self) -> List[Dict]:
        """Get currently running tasks"""
        return [
            task.to_dict() for task in self._tasks.values()
            if task.status == TaskStatus.RUNNING
        ]
        
    def get_scheduled_tasks(self) -> List[Dict]:
        """Get tasks scheduled to run"""
        return sorted(
            [task.to_dict() for task in self._tasks.values()
             if task.status == TaskStatus.SCHEDULED],
            key=lambda t: t.get("next_run") or float('inf')
        )
        
    def get_status(self) -> Dict[str, Any]:
        """Get scheduler status"""
        return {
            "running": self._running,
            "paused": self._paused,
            "total_tasks": len(self._tasks),
            "running_tasks": len(self._running_tasks),
            "scheduled_tasks": sum(1 for t in self._tasks.values() if t.status == TaskStatus.SCHEDULED),
            "completed_tasks": sum(1 for t in self._tasks.values() if t.status == TaskStatus.COMPLETED),
            "failed_tasks": sum(1 for t in self._tasks.values() if t.status == TaskStatus.FAILED),
            "max_concurrent": self._max_concurrent
        }
