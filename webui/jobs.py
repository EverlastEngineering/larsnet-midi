"""
Job Queue System

Manages asynchronous execution of long-running operations (separation, MIDI conversion, etc.)
with status tracking and real-time progress updates.

Architecture: Functional Core, Imperative Shell
- Pure functions for job state management
- Imperative shell handles threading and I/O
"""

import uuid
import threading
import time
import queue
import sys
import io
from datetime import datetime
from typing import Dict, Callable, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum
import traceback


class JobStatus(Enum):
    """Job execution status"""
    QUEUED = "Queued"
    RUNNING = "Initlializing"
    COMPLETED = "Completed"
    FAILED = "Failed"
    CANCELLED = "Cancelled"


@dataclass
class JobLog:
    """Single log entry from job execution"""
    timestamp: datetime
    level: str  # 'info', 'warning', 'error'
    message: str


@dataclass
class Job:
    """
    Represents a single job in the queue.
    
    Attributes:
        id: Unique job identifier
        operation: Operation name (e.g., 'separate', 'stems-to-midi')
        project_id: Associated project ID
        status: Current job status
        status_detail: Detailed status message (e.g., 'Processing kick')
        progress: Progress percentage (0-100)
        logs: List of log entries
        result: Result data (if completed)
        error: Error message (if failed)
        created_at: Job creation timestamp
        started_at: Job start timestamp
        completed_at: Job completion timestamp
    """
    id: str
    operation: str
    project_id: Optional[int]
    status: JobStatus = JobStatus.QUEUED
    status_detail: str = ""
    progress: int = 0
    logs: List[JobLog] = field(default_factory=list)
    result: Optional[Any] = None
    error: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    def add_log(self, level: str, message: str):
        """Add a log entry to this job"""
        self.logs.append(JobLog(
            timestamp=datetime.now(),
            level=level,
            message=message
        ))
    
    def to_dict(self) -> dict:
        """Convert job to dictionary for API responses"""
        return {
            'id': self.id,
            'operation': self.operation,
            'project_id': self.project_id,
            'status': self.status.value,
            'status_detail': self.status_detail,
            'progress': self.progress,
            'logs': [
                {
                    'timestamp': log.timestamp.isoformat(),
                    'level': log.level,
                    'message': log.message
                }
                for log in self.logs
            ],
            'result': self.result,
            'error': self.error,
            'created_at': self.created_at.isoformat(),
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None
        }


class StdoutCapture:
    """
    Context manager to capture stdout/stderr and add to job logs.
    
    Usage:
        with StdoutCapture(job):
            print("This will be captured")  # Added to job.logs
    """
    def __init__(self, job: Job):
        self.job = job
        self.stdout_buffer = io.StringIO()
        self.stderr_buffer = io.StringIO()
        self.old_stdout = None
        self.old_stderr = None
    
    def __enter__(self):
        self.old_stdout = sys.stdout
        self.old_stderr = sys.stderr
        
        # Create wrapper objects for stdout and stderr that write to both job logs and console
        sys.stdout = StdoutWrapper(self.stdout_buffer, self.job, 'info', self.old_stdout)
        sys.stderr = StdoutWrapper(self.stderr_buffer, self.job, 'error', self.old_stderr)
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Flush any remaining output
        stdout_remaining = self.stdout_buffer.getvalue()
        if stdout_remaining.strip():
            for line in stdout_remaining.strip().split('\n'):
                if line.strip():
                    self.job.add_log('info', line.strip())
        
        stderr_remaining = self.stderr_buffer.getvalue()
        if stderr_remaining.strip():
            for line in stderr_remaining.strip().split('\n'):
                if line.strip():
                    self.job.add_log('error', line.strip())
        
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr
        return False


class StdoutWrapper:
    """Wrapper for stdout/stderr that captures output line-by-line"""
    def __init__(self, buffer: io.StringIO, job: Job, level: str, original_stream=None):
        self.buffer = buffer
        self.job = job
        self.level = level
        self.original_stream = original_stream
    
    def write(self, text):
        """Write to buffer and flush complete lines to job logs"""
        if not text:
            return
        
        # Write to original stream (console) as well
        if self.original_stream:
            self.original_stream.write(text)
            self.original_stream.flush()
        
        self.buffer.write(text)
        
        # Process complete lines
        # if \n or \r in text
        if '\n' in text or '\r' in text:
            content = self.buffer.getvalue()

            # make lines split by \n or by \r
            if '\r' in content:
                lines = content.split('\r')
            else:
                lines = content.split('\n')
            # lines = content.replace('\r', '\n').split('\n')
            
            # Log all complete lines
            for line in lines[:-1]:
                if line.strip():
                    self.job.add_log(self.level, line.strip())
                    
                    # Check for progress updates in format "Progress: X%"
                    if 'Progress:' in line and '%' in line:
                        try:
                            # Extract percentage (e.g., "Progress: 45.2%" -> 45)
                            progress_str = line.split('Progress:')[1].split('%')[0].strip()
                            progress = int(float(progress_str))
                            self.job.progress = max(0, min(100, progress))
                        except (ValueError, IndexError):
                            pass  # Ignore malformed progress lines
                    
                    # Check for other status indicators
                    if 'Status Update: ' in line:
                        status_msg = line.split('Status Update: ')[1].strip()
                        self.job.status_detail = status_msg
                    # Check for stem processing messages from tqdm (e.g., "kick pretrained_kick_unet")
                    elif any(stem in line.lower() for stem in ['kick', 'snare', 'toms', 'hihat', 'cymbals']):
                        # Only update if it's a processing message, not just mentioning the stem
                        if 'pretrained' in line.lower() or 'processing' in line.lower():
                            # Extract stem name
                            for stem in ['kick', 'snare', 'toms', 'hihat', 'cymbals']:
                                if stem in line.lower():
                                    self.job.status_detail = f"Stem Splitting - {stem.capitalize()}"
                                    break
           
            # Keep incomplete line in buffer
            self.buffer = io.StringIO()
            if lines[-1]:
                self.buffer.write(lines[-1])
    
    def flush(self):
        """Flush method for compatibility"""
        pass


class JobQueue:
    """
    Thread-safe job queue for managing asynchronous operations.
    
    Provides:
    - Job submission and execution
    - Status tracking
    - Real-time log streaming
    - Concurrent job limiting
    """
    
    def __init__(self, max_concurrent: int = 1):
        """
        Initialize job queue.
        
        Args:
            max_concurrent: Maximum number of jobs to run concurrently
        """
        self.max_concurrent = max_concurrent
        self.jobs: Dict[str, Job] = {}
        self.job_lock = threading.Lock()
        self.worker_threads: List[threading.Thread] = []
        self.running = False
        
    def start(self):
        """Start worker threads"""
        if self.running:
            return
            
        self.running = True
        for i in range(self.max_concurrent):
            thread = threading.Thread(target=self._worker, daemon=True, name=f'JobWorker-{i}')
            thread.start()
            self.worker_threads.append(thread)
    
    def stop(self):
        """Stop worker threads"""
        self.running = False
        for thread in self.worker_threads:
            thread.join(timeout=5)
        self.worker_threads.clear()
    
    def submit(
        self,
        operation: str,
        func: Callable,
        project_id: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Submit a new job to the queue.
        
        Args:
            operation: Operation name (e.g., 'separate', 'stems-to-midi')
            func: Function to execute
            project_id: Associated project ID
            **kwargs: Arguments to pass to func
        
        Returns:
            Job ID
        """
        job_id = str(uuid.uuid4())
        job = Job(
            id=job_id,
            operation=operation,
            project_id=project_id
        )
        job.add_log('info', f'Job queued: {operation}')
        
        with self.job_lock:
            self.jobs[job_id] = job
        
        # Store function and args for worker to execute
        job._func = func
        job._kwargs = kwargs
        
        return job_id
    
    def get_job(self, job_id: str) -> Optional[Job]:
        """Get job by ID"""
        with self.job_lock:
            return self.jobs.get(job_id)
    
    def get_all_jobs(self) -> List[Job]:
        """Get all jobs"""
        with self.job_lock:
            return list(self.jobs.values())
    
    def get_project_jobs(self, project_id: int) -> List[Job]:
        """Get all jobs for a specific project"""
        with self.job_lock:
            return [job for job in self.jobs.values() if job.project_id == project_id]
    
    def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a job.
        
        Note: This only marks the job as cancelled. If it's already running,
        it will continue until the function completes.
        
        Args:
            job_id: Job ID to cancel
            
        Returns:
            True if job was cancelled, False if not found or already completed
        """
        job = self.get_job(job_id)
        if job is None:
            return False
        
        if job.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
            return False
        
        with self.job_lock:
            job.status = JobStatus.CANCELLED
            job.completed_at = datetime.now()
            job.add_log('warning', 'Job cancelled by user')
        
        return True
    
    def _worker(self):
        """Worker thread that processes jobs"""
        while self.running:
            # Find next queued job
            job_to_run = None
            
            with self.job_lock:
                for job in self.jobs.values():
                    if job.status == JobStatus.QUEUED:
                        job.status = JobStatus.RUNNING
                        job.started_at = datetime.now()
                        job.add_log('info', f'Job started: {job.operation}')
                        job_to_run = job
                        break
            
            if job_to_run is None:
                # No jobs to run, sleep and check again
                time.sleep(0.5)
                continue
            
            # Execute job with stdout/stderr capture
            try:
                job_to_run.add_log('info', 'Executing operation...')
                
                # Capture stdout/stderr and add to job logs
                with StdoutCapture(job_to_run):
                    result = job_to_run._func(**job_to_run._kwargs)
                
                with self.job_lock:
                    if job_to_run.status == JobStatus.CANCELLED:
                        job_to_run.add_log('warning', 'Job was cancelled')
                    else:
                        job_to_run.status = JobStatus.COMPLETED
                        job_to_run.progress = 100
                        job_to_run.result = result
                        job_to_run.completed_at = datetime.now()
                        job_to_run.add_log('info', 'Job completed successfully')
                        
            except Exception as e:
                error_msg = str(e)
                error_trace = traceback.format_exc()
                
                with self.job_lock:
                    job_to_run.status = JobStatus.FAILED
                    job_to_run.error = error_msg
                    job_to_run.completed_at = datetime.now()
                    job_to_run.add_log('error', f'Job failed: {error_msg}')
                    job_to_run.add_log('error', f'Traceback:\n{error_trace}')


# Global job queue instance
_job_queue: Optional[JobQueue] = None


def get_job_queue() -> JobQueue:
    """Get the global job queue instance"""
    global _job_queue
    if _job_queue is None:
        _job_queue = JobQueue()
        _job_queue.start()
    return _job_queue


def shutdown_job_queue():
    """Shutdown the global job queue"""
    global _job_queue
    if _job_queue is not None:
        _job_queue.stop()
        _job_queue = None
