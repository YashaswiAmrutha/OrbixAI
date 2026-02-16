import unittest
import sys
import os
from unittest.mock import patch, MagicMock
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend')))

from backend.intent_workflow.workflow_executor import WorkflowExecutor, WorkflowTask


class TestWorkflowTaskDataclass(unittest.TestCase):

    def test_create_task_with_defaults(self):
        task = WorkflowTask(name="task_a", function=lambda: {}, required_params=[])
        self.assertEqual(task.name, "task_a")
        self.assertEqual(task.on_error, "stop")

    def test_create_task_with_continue(self):
        task = WorkflowTask(name="task_b", function=lambda: {}, required_params=[], on_error="continue")
        self.assertEqual(task.on_error, "continue")

    def test_task_stores_function(self):
        fn = lambda: {"result": True}
        task = WorkflowTask(name="t", function=fn, required_params=[])
        self.assertIs(task.function, fn)

    def test_task_stores_required_params(self):
        task = WorkflowTask(name="t", function=lambda: {}, required_params=["a", "b"])
        self.assertEqual(task.required_params, ["a", "b"])


class TestWorkflowExecutorInit(unittest.TestCase):

    def test_executor_creates_empty_workflows(self):
        executor = WorkflowExecutor()
        self.assertEqual(len(executor.workflows), 0)

    def test_executor_has_task_results(self):
        executor = WorkflowExecutor()
        self.assertIsInstance(executor.task_results, dict)


class TestWorkflowRegistration(unittest.TestCase):

    def test_register_single_workflow(self):
        executor = WorkflowExecutor()
        task = WorkflowTask(name="t", function=lambda: {}, required_params=[])
        executor.register_workflow("intent_a", [task])
        self.assertIn("intent_a", executor.workflows)

    def test_register_multiple_workflows(self):
        executor = WorkflowExecutor()
        t1 = WorkflowTask(name="t1", function=lambda: {}, required_params=[])
        t2 = WorkflowTask(name="t2", function=lambda: {}, required_params=[])
        executor.register_workflow("a", [t1])
        executor.register_workflow("b", [t2])
        self.assertEqual(len(executor.workflows), 2)

    def test_register_empty_workflow(self):
        executor = WorkflowExecutor()
        executor.register_workflow("empty", [])
        self.assertIn("empty", executor.workflows)
        self.assertEqual(len(executor.workflows["empty"]), 0)

    def test_overwrite_workflow(self):
        executor = WorkflowExecutor()
        t1 = WorkflowTask(name="t1", function=lambda: {}, required_params=[])
        t2 = WorkflowTask(name="t2", function=lambda: {}, required_params=[])
        executor.register_workflow("x", [t1])
        executor.register_workflow("x", [t2])
        self.assertEqual(executor.workflows["x"][0].name, "t2")


class TestWorkflowExecution(unittest.TestCase):

    def test_execute_single_task(self):
        executor = WorkflowExecutor()
        task = WorkflowTask(name="task_ok", function=lambda: {"done": True}, required_params=[])
        executor.register_workflow("intent", [task])
        result = executor.execute("intent", {})
        self.assertTrue(result["success"])
        self.assertIn("task_ok", result["tasks_executed"])

    def test_execute_returns_results(self):
        executor = WorkflowExecutor()
        task = WorkflowTask(name="task_ok", function=lambda: {"key": "value"}, required_params=[])
        executor.register_workflow("intent", [task])
        result = executor.execute("intent", {})
        self.assertEqual(result["results"]["task_ok"]["key"], "value")

    def test_execute_unregistered_intent(self):
        executor = WorkflowExecutor()
        result = executor.execute("missing", {})
        self.assertFalse(result["success"])
        self.assertIn("error", result)

    def test_execute_sequential_tasks(self):
        executor = WorkflowExecutor()
        t1 = WorkflowTask(name="first", function=lambda: {"output": "from_first"}, required_params=[])
        t2 = WorkflowTask(name="second", function=lambda output=None: {"received": output}, required_params=[])
        executor.register_workflow("seq", [t1, t2])
        result = executor.execute("seq", {})
        self.assertTrue(result["success"])
        self.assertEqual(len(result["tasks_executed"]), 2)

    def test_execute_empty_workflow(self):
        executor = WorkflowExecutor()
        executor.register_workflow("noop", [])
        result = executor.execute("noop", {})
        self.assertTrue(result["success"])
        self.assertEqual(result["tasks_executed"], [])


class TestWorkflowErrorHandling(unittest.TestCase):

    def test_task_error_stop_policy(self):
        executor = WorkflowExecutor()

        def fail():
            raise RuntimeError("boom")

        t1 = WorkflowTask(name="fail_task", function=fail, required_params=[], on_error="stop")
        t2 = WorkflowTask(name="skip_task", function=lambda: {}, required_params=[])
        executor.register_workflow("e", [t1, t2])
        result = executor.execute("e", {})
        self.assertFalse(result["success"])
        self.assertIn("fail_task", result["tasks_failed"])
        self.assertNotIn("skip_task", result["tasks_executed"])

    def test_task_error_continue_policy(self):
        executor = WorkflowExecutor()

        def fail():
            raise RuntimeError("boom")

        t1 = WorkflowTask(name="fail_task", function=fail, required_params=[], on_error="continue")
        t2 = WorkflowTask(name="next_task", function=lambda: {"ok": True}, required_params=[])
        executor.register_workflow("e", [t1, t2])
        result = executor.execute("e", {})
        self.assertFalse(result["success"])
        self.assertIn("fail_task", result["tasks_failed"])
        self.assertIn("next_task", result["tasks_executed"])

    def test_error_message_recorded(self):
        executor = WorkflowExecutor()

        def fail():
            raise ValueError("specific error")

        t = WorkflowTask(name="bad", function=fail, required_params=[])
        executor.register_workflow("e", [t])
        result = executor.execute("e", {})
        self.assertIn("bad", result["errors"])
        self.assertIn("specific error", result["errors"]["bad"])


class TestWorkflowParameterPassing(unittest.TestCase):

    def test_parameters_forwarded(self):
        executor = WorkflowExecutor()
        received = {}

        def capture(email=None):
            received["email"] = email
            return {}

        t = WorkflowTask(name="cap", function=capture, required_params=["email"])
        executor.register_workflow("p", [t])
        executor.execute("p", {"email": "a@b.com"})
        self.assertEqual(received["email"], "a@b.com")

    def test_first_task_results_available_to_second(self):
        executor = WorkflowExecutor()

        def step1():
            return {"meet_link": "https://meet.google.com/abc"}

        def step2(meet_link=None):
            return {"link_received": meet_link}

        t1 = WorkflowTask(name="s1", function=step1, required_params=[])
        t2 = WorkflowTask(name="s2", function=step2, required_params=[])
        executor.register_workflow("chain", [t1, t2])
        result = executor.execute("chain", {})
        self.assertTrue(result["success"])

    def test_missing_required_param_logged(self):
        executor = WorkflowExecutor()
        t = WorkflowTask(name="t", function=lambda x=None: {}, required_params=["x"])
        executor.register_workflow("p", [t])
        result = executor.execute("p", {})
        self.assertTrue(result["success"])


if __name__ == "__main__":
    unittest.main()
