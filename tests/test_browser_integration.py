import os
import socket
import subprocess
import sys
import time
import unittest
from urllib.request import urlopen


def _port_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.3)
        return sock.connect_ex((host, port)) == 0


class TestBrowserIntegration(unittest.TestCase):
    server_proc = None
    base_url = "http://127.0.0.1:5000"

    @classmethod
    def setUpClass(cls):
        try:
            from playwright.sync_api import sync_playwright  # noqa: F401
        except Exception as exc:
            raise unittest.SkipTest(f"Playwright not available: {exc}")

        if _port_open("127.0.0.1", 5000):
            raise unittest.SkipTest("Port 5000 is already in use; cannot run browser integration tests safely.")

        env = os.environ.copy()
        env["FLASK_DEBUG"] = "0"
        cls.server_proc = subprocess.Popen(
            [sys.executable, "app.py"],
            cwd=os.getcwd(),
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        deadline = time.time() + 20
        while time.time() < deadline:
            try:
                with urlopen(cls.base_url, timeout=0.5) as resp:
                    if resp.status == 200:
                        return
            except Exception:
                time.sleep(0.2)

        if cls.server_proc is not None:
            cls.server_proc.terminate()
            cls.server_proc.wait(timeout=5)
        raise RuntimeError("Server did not start in time for browser integration tests.")

    @classmethod
    def tearDownClass(cls):
        if cls.server_proc is not None:
            cls.server_proc.terminate()
            try:
                cls.server_proc.wait(timeout=5)
            except Exception:
                cls.server_proc.kill()

    def _start_two_player_game(self, p1, p2):
        p1.goto(self.base_url, wait_until="networkidle")
        p2.goto(self.base_url, wait_until="networkidle")

        p1.fill("#lobby-name", "Alice")
        p1.evaluate("createRoom()")
        p1.wait_for_selector("#lobby-waiting", state="visible")
        code = p1.inner_text("#lobby-code").strip()
        self.assertTrue(len(code) == 4)

        p2.fill("#lobby-name", "Bob")
        p2.fill("#join-code", code)
        p2.evaluate("peekRoom()")
        p2.wait_for_selector("#lobby-seat-picker", state="visible")
        p2.click(".seat-pick-row.open")
        p2.wait_for_selector("#lobby-waiting", state="visible")

        p1.evaluate("startGame()")
        p1.wait_for_function("!document.getElementById('lobby-overlay').classList.contains('active')")
        p2.wait_for_function("!document.getElementById('lobby-overlay').classList.contains('active')")
        return code

    def test_create_join_start_and_chat(self):
        try:
            from playwright.sync_api import sync_playwright
        except Exception as exc:
            raise unittest.SkipTest(f"Playwright not available: {exc}")

        with sync_playwright() as pw:
            try:
                browser = pw.chromium.launch(headless=True)
            except Exception as exc:
                raise unittest.SkipTest(f"Chromium not installed for Playwright: {exc}")

            try:
                ctx1 = browser.new_context()
                ctx2 = browser.new_context()
                p1 = ctx1.new_page()
                p2 = ctx2.new_page()

                self._start_two_player_game(p1, p2)

                p2.evaluate("openChat()")
                p1.evaluate("openChat()")
                p1.fill("#chat-input", "hello from e2e")
                p1.evaluate("sendChat()")
                p2.wait_for_selector(".chat-msg")
                self.assertIn("hello from e2e", p2.inner_text("#chat-messages"))
            finally:
                browser.close()

    def _session(self, page):
        return page.evaluate(
            "(() => { try { return JSON.parse(localStorage.getItem('klaverjas_session') || '{}'); } catch (e) { return {}; } })()"
        )

    def _play_until_next_round_banner(self, pages, host_page, timeout=240):
        """Drive both humans through a whole round: declare whatever is
        offered, play the first legal card whenever asked, until the host
        sees the next-round banner."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            if host_page.evaluate("document.getElementById('nextround-banner').classList.contains('active')"):
                return
            for page in pages:
                try:
                    if page.evaluate("document.getElementById('bid-overlay').classList.contains('active')"):
                        page.click("#bid-buttons button", timeout=2000)
                        continue
                    legal = page.query_selector(".card-face.legal")
                    if legal is not None:
                        legal.click(timeout=2000)
                except Exception:
                    pass  # the table re-rendered under us; try again
            time.sleep(0.3)
        raise AssertionError("round did not finish in time")

    def test_network_drop_mid_game_recovers_without_lobby(self):
        try:
            from playwright.sync_api import sync_playwright
        except Exception as exc:
            raise unittest.SkipTest(f"Playwright not available: {exc}")

        with sync_playwright() as pw:
            try:
                browser = pw.chromium.launch(headless=True)
            except Exception as exc:
                raise unittest.SkipTest(f"Chromium not installed for Playwright: {exc}")

            try:
                ctx1 = browser.new_context()
                ctx2 = browser.new_context()
                p1 = ctx1.new_page()
                p2 = ctx2.new_page()

                code = self._start_two_player_game(p1, p2)
                before = self._session(p2)
                self.assertEqual(before.get("code"), code)
                self.assertTrue(before.get("token"))

                # Bob's network drops. Alice sees the pause once the server
                # notices (at the latest after the ping timeout).
                ctx2.set_offline(True)
                p1.wait_for_function(
                    "document.getElementById('paused-overlay').classList.contains('active')",
                    timeout=90000,
                )
                self.assertIn("Bob", p1.inner_text("#paused-msg"))

                # Bob is back: the client rejoins with its token, the pause
                # clears for Alice, and Bob never sees the lobby form.
                ctx2.set_offline(False)
                p1.wait_for_function(
                    "!document.getElementById('paused-overlay').classList.contains('active')",
                    timeout=90000,
                )
                p2.wait_for_function(
                    "!document.getElementById('lobby-overlay').classList.contains('active')"
                    " && !document.getElementById('paused-overlay').classList.contains('active')",
                    timeout=90000,
                )
                after = self._session(p2)
                self.assertEqual(after.get("code"), code)
                self.assertEqual(after.get("token"), before.get("token"))
                leave_display = p2.evaluate("getComputedStyle(document.getElementById('leave-game-btn')).display")
                self.assertNotEqual(leave_display, "none")
            finally:
                browser.close()

    def test_second_tab_takes_over_seat_and_can_be_taken_back(self):
        try:
            from playwright.sync_api import sync_playwright
        except Exception as exc:
            raise unittest.SkipTest(f"Playwright not available: {exc}")

        with sync_playwright() as pw:
            try:
                browser = pw.chromium.launch(headless=True)
            except Exception as exc:
                raise unittest.SkipTest(f"Chromium not installed for Playwright: {exc}")

            try:
                ctx1 = browser.new_context()
                ctx2 = browser.new_context()
                p1 = ctx1.new_page()
                p2 = ctx2.new_page()
                self._start_two_player_game(p1, p2)

                # A second tab in Bob's browser shares the stored token: it
                # takes the seat over and the first tab goes dormant.
                p3 = ctx2.new_page()
                p3.goto(self.base_url, wait_until="networkidle")
                p3.wait_for_function("!document.getElementById('lobby-overlay').classList.contains('active')")
                p2.wait_for_function("document.getElementById('superseded-overlay').classList.contains('active')")
                # Alice never saw a pause.
                self.assertFalse(p1.evaluate("document.getElementById('paused-overlay').classList.contains('active')"))

                # Bob takes the seat back from the first tab.
                p2.evaluate("takeOverSeat()")
                p2.wait_for_function(
                    "!document.getElementById('superseded-overlay').classList.contains('active')"
                    " && !document.getElementById('lobby-overlay').classList.contains('active')"
                    " && !document.getElementById('paused-overlay').classList.contains('active')"
                )
                p3.wait_for_function("document.getElementById('superseded-overlay').classList.contains('active')")
            finally:
                browser.close()

    def test_host_reload_between_rounds_restores_next_round_button(self):
        try:
            from playwright.sync_api import sync_playwright
        except Exception as exc:
            raise unittest.SkipTest(f"Playwright not available: {exc}")

        with sync_playwright() as pw:
            try:
                browser = pw.chromium.launch(headless=True)
            except Exception as exc:
                raise unittest.SkipTest(f"Chromium not installed for Playwright: {exc}")

            try:
                ctx1 = browser.new_context()
                ctx2 = browser.new_context()
                p1 = ctx1.new_page()
                p2 = ctx2.new_page()
                self._start_two_player_game(p1, p2)
                self._play_until_next_round_banner([p1, p2], host_page=p1)

                # The host reloads while the game waits for "Next round".
                p1.reload(wait_until="networkidle")
                p1.wait_for_function("!document.getElementById('lobby-overlay').classList.contains('active')")
                p1.wait_for_function("document.getElementById('nextround-banner').classList.contains('active')")
                btn_display = p1.evaluate("getComputedStyle(document.getElementById('nextround-btn')).display")
                self.assertNotEqual(btn_display, "none")
                # The button works: the next deal hides the banner again.
                p1.evaluate("nextRound()")
                p1.wait_for_function("!document.getElementById('nextround-banner').classList.contains('active')")
                p2.wait_for_function("!document.getElementById('nextround-banner').classList.contains('active')")
            finally:
                browser.close()

    def test_reload_auto_reconnect_restores_active_game(self):
        try:
            from playwright.sync_api import sync_playwright
        except Exception as exc:
            raise unittest.SkipTest(f"Playwright not available: {exc}")

        with sync_playwright() as pw:
            try:
                browser = pw.chromium.launch(headless=True)
            except Exception as exc:
                raise unittest.SkipTest(f"Chromium not installed for Playwright: {exc}")

            try:
                ctx1 = browser.new_context()
                ctx2 = browser.new_context()
                p1 = ctx1.new_page()
                p2 = ctx2.new_page()

                code = self._start_two_player_game(p1, p2)

                # Ensure session was written before reload.
                session_code = p2.evaluate("(() => { const s = JSON.parse(localStorage.getItem('klaverjas_session') || '{}'); return s.code || ''; })()")
                self.assertEqual(session_code, code)

                p2.reload(wait_until="networkidle")

                # Auto reconnect should hide lobby and restore game controls.
                p2.wait_for_function("!document.getElementById('lobby-overlay').classList.contains('active')")
                leave_display = p2.evaluate("getComputedStyle(document.getElementById('leave-game-btn')).display")
                self.assertNotEqual(leave_display, "none")
                paused_active = p2.evaluate("document.getElementById('paused-overlay').classList.contains('active')")
                self.assertFalse(paused_active)
            finally:
                browser.close()


if __name__ == "__main__":
    unittest.main()
