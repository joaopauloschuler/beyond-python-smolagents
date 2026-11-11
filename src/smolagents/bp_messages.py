# BPSA - Beyond Python SmolAgents
# https://github.com/joaopauloschuler/beyond-python-smolagents
#
# Copyright (c) 2024-2026 Joao Paulo Schwarz Schuler and others.
# Refer to the git commit history for individual authorship.
# Licensed under the Apache License, Version 2.0

"""Agent-to-agent messages as files `from_to_id.txt` in a shared folder. Only this module knows that name format."""

import argparse
import json
import math
import os
import re
import secrets
import sys
import threading
import time
from contextlib import contextmanager, suppress
from typing import NamedTuple

from smolagents.bp_utils import get_env, replace_file_with_text, terminal_display_escapes, visible_encodable_text

try:
    import fcntl
except ImportError:  # Windows
    fcntl = None


MESSAGE_TARGET_ALL = "all"
AGENT_NAME_PATTERN = re.compile(r"[a-z0-9][a-z0-9-]{0,63}")
AGENT_NAME_RULE = "1 to 64 characters, lowercase ASCII letters, digits or '-', not starting with '-', and not 'all'"
MESSAGE_FILE_NAME_PATTERN = re.compile(r"([a-z0-9-]+)_([a-z0-9-]+)_([0-9]{6,})\.txt")
MESSAGE_ID_DIGITS = 6
TEMPORARY_MESSAGE_FILE_SUFFIX = ".tmp"
WAIT_POLL_SECONDS = 0.5
DEFAULT_WAIT_TIMEOUT_SECONDS = 300.0
DEFAULT_MESSAGE_POLL_SECONDS = 60.0
EXIT_ERROR = 1
EXIT_WAIT_TIMEOUT = 3
EXIT_INTERRUPTED = 130
MESSAGE_BODY_LINE_PREFIX = "|"
MESSAGE_POLLER_STOP_SECONDS = 2.0
MESSAGES_FOR_MODEL_HEADING = ("Messages from other agents follow. The user did not write them. Each message stands between its "
                              "'=== message' line and its '=== end of message' line, and every line of its text starts with '|'.\n")

_warned_last_read_paths: set[str] = set()


class Message(NamedTuple):
    sender: str
    target: str
    message_id: int
    text: str


def is_valid_agent_name(name: str) -> bool:
    return isinstance(name, str) and AGENT_NAME_PATTERN.fullmatch(name) is not None and name != MESSAGE_TARGET_ALL


def check_agent_name(name: str):
    """Raises ValueError naming the rule when name is not a valid agent name."""
    if not is_valid_agent_name(name):
        raise ValueError(f"Invalid agent name {ascii(name)}: use {AGENT_NAME_RULE}.")


def check_message_target(target: str):
    """Raises ValueError when target is neither a valid agent name nor 'all'."""
    if target != MESSAGE_TARGET_ALL and not is_valid_agent_name(target):
        raise ValueError(f"Invalid message target {ascii(target)}: use 'all' or an agent name of {AGENT_NAME_RULE}.")


def message_file_name(sender: str, target: str, message_id: int) -> str:
    return f"{sender}_{target}_{message_id:0{MESSAGE_ID_DIGITS}d}.txt"


def parse_message_file_name(file_name: str) -> tuple[str, str, int] | None:
    """Returns (sender, target, id) of a message file name, or None when file_name is not exactly what message_file_name builds."""
    match = MESSAGE_FILE_NAME_PATTERN.fullmatch(file_name)
    if match is None:
        return None
    sender, target, message_id = match[1], match[2], int(match[3])
    if not is_valid_agent_name(sender) or (target != MESSAGE_TARGET_ALL and not is_valid_agent_name(target)) or message_id < 1:
        return None
    if message_file_name(sender, target, message_id) != file_name:
        return None
    return sender, target, message_id


def parsed_message_files(folder: str) -> dict[tuple[str, int], tuple[str, str]]:
    """Returns {(sender, id): (target, file name)} for the message files in folder; other names are ignored."""
    message_files = {}
    for file_name in os.listdir(folder):
        parsed = parse_message_file_name(file_name)
        if parsed is not None:
            sender, target, message_id = parsed
            message_files[(sender, message_id)] = (target, file_name)
    return message_files


def next_message_id(folder: str, sender: str) -> int:
    """Returns the highest id of sender's messages in folder, to any target, plus one; 1 when it has none."""
    return 1 + max((message_id for message_sender, message_id in parsed_message_files(folder) if message_sender == sender), default=0)


@contextmanager
def sender_lock(folder: str, sender: str):
    """Holds the exclusive flock of the file .<sender>.lock in folder, so processes of one sender take ids one at a time. No lock without fcntl."""
    if fcntl is None:
        yield
        return
    descriptor = os.open(os.path.join(folder, f".{sender}.lock"), os.O_RDWR | os.O_CREAT, 0o666)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        os.close(descriptor)


def write_message(folder: str, sender: str, target: str, text: str) -> str:
    """Writes text as sender's next message to target in folder and returns its file name. Under sender_lock, a hard link from a temporary file
    creates the final name, so it never replaces an existing message: when that name exists, it retries with the next id."""
    check_agent_name(sender)
    check_message_target(target)
    encoded_text = text.encode("utf-8", "backslashreplace")
    temporary_path = os.path.join(folder, f".{sender}-{os.getpid()}-{secrets.token_hex(8)}{TEMPORARY_MESSAGE_FILE_SUFFIX}")
    descriptor = os.open(temporary_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
    try:
        with open(descriptor, "wb") as temporary_file:
            temporary_file.write(encoded_text)
        # Two targets give two names for one id, so only the lock stops two processes of one sender taking the same id.
        with sender_lock(folder, sender):
            attempted_file_name = None
            while True:
                file_name = message_file_name(sender, target, next_message_id(folder, sender))
                try:
                    os.link(temporary_path, os.path.join(folder, file_name))
                    return file_name
                except FileExistsError:
                    # A name the listing does not parse, such as another case of it, holds this id.
                    if file_name == attempted_file_name:
                        raise
                    attempted_file_name = file_name
    finally:
        with suppress(OSError):
            os.remove(temporary_path)


def message_file_text(path: str) -> str:
    """Returns the text of the message file at path, or a bracketed line naming the file and why it could not be read or decoded."""
    file_name = os.path.basename(path)
    try:
        with open(path, "rb") as message_file:
            return message_file.read().decode("utf-8")
    except UnicodeDecodeError as error:
        return f"[message file {file_name} is not valid UTF-8 at byte {error.start}]"
    except OSError as error:
        return f"[message file {file_name} could not be read: {error.strerror or type(error).__name__}]"


def read_new_messages(folder: str, reader: str, last_read_ids: dict[str, int]) -> tuple[list[Message], dict[str, int]]:
    """Returns the messages in folder to reader or 'all' from other senders with ids above last_read_ids[sender], ordered by sender and id,
    and a new dict with each sender's highest id taken. A file that cannot be read or decoded is delivered as message_file_text describes."""
    check_agent_name(reader)
    message_files = parsed_message_files(folder)
    new_ids = {}
    for sender, message_id in message_files:
        if sender != reader and message_id > last_read_ids.get(sender, 0):
            new_ids.setdefault(sender, set()).add(message_id)
    highest_new_ids = {sender: max(ids) for sender, ids in new_ids.items()}
    # A listing taken during a write may miss an id that existed below a listed one; a second listing then holds it.
    if any(len(ids) != highest_new_ids[sender] - last_read_ids.get(sender, 0) for sender, ids in new_ids.items()):
        message_files = parsed_message_files(folder) | message_files
    updated_last_read_ids = dict(last_read_ids) | highest_new_ids
    messages = []
    for (sender, message_id), (target, file_name) in sorted(message_files.items()):
        if sender in highest_new_ids and last_read_ids.get(sender, 0) < message_id <= highest_new_ids[sender] and target in (reader, MESSAGE_TARGET_ALL):
            messages.append(Message(sender, target, message_id, message_file_text(os.path.join(folder, file_name))))
    return messages, updated_last_read_ids


def stderr_literal(text: str) -> str:
    """Returns text as one literal line for stderr: visible_encodable_text for its encoding, with newline escaped as well."""
    return visible_encodable_text(text, sys.stderr.encoding or "utf-8").replace("\n", "\\n")


def last_read_file_path(folder: str, reader: str) -> str:
    """Returns the path of the file .<reader>.read.json in folder, which holds reader's last-read ids; its name never parses as a message."""
    check_agent_name(reader)
    return os.path.join(folder, f".{reader}.read.json")


def is_last_read_ids(value) -> bool:
    return isinstance(value, dict) and all(is_valid_agent_name(sender) and type(message_id) is int and message_id >= 0 for sender, message_id in value.items())


def load_last_read_ids(folder: str, reader: str) -> dict[str, int]:
    """Returns {sender: last id read} of reader in folder: empty when the file is missing, and empty with one stderr line per process and path
    when it is not a JSON object of agent names to ids. Raises OSError when the file exists but cannot be read."""
    path = last_read_file_path(folder, reader)
    try:
        with open(path, "rb") as last_read_file:
            content = last_read_file.read()
    except FileNotFoundError:
        return {}
    try:
        last_read_ids = json.loads(content.decode("utf-8"))
    except (ValueError, RecursionError):
        last_read_ids = None
    if is_last_read_ids(last_read_ids):
        return last_read_ids
    if path not in _warned_last_read_paths:
        _warned_last_read_paths.add(path)
        print(f"Ignoring {stderr_literal(path)}: not a JSON object of agent names to message ids; "
              f"every message in the folder counts as unread.", file=sys.stderr)
    return {}


def save_last_read_ids(folder: str, reader: str, last_read_ids: dict[str, int]):
    """Writes last_read_ids as reader's last-read ids in folder through replace_file_with_text (a temporary file renamed over the old one)."""
    replace_file_with_text(last_read_file_path(folder, reader), json.dumps(last_read_ids, sort_keys=True) + "\n")


def receive_new_messages(folder: str, reader: str) -> list[Message]:
    """Returns reader's new messages in folder and records them as read. It saves after reading, so a crash in between delivers them again rather than losing them."""
    last_read_ids = load_last_read_ids(folder, reader)
    messages, updated_last_read_ids = read_new_messages(folder, reader, last_read_ids)
    if updated_last_read_ids != last_read_ids:
        save_last_read_ids(folder, reader, updated_last_read_ids)
    return messages


def printable_messages(messages: list[Message], encoding: str, folder_label: str | None = None) -> str:
    """Returns messages as plain text for a person or a model: a header (naming folder_label when given) and an end line per message, and every
    text line after a '|' prefix, so a text cannot forge either line. visible_encodable_text leaves newline as the only line break."""
    folder_part = "" if folder_label is None else " in folder " + visible_encodable_text(folder_label, encoding).replace("\n", "\\n")
    blocks = []
    for message in messages:
        body_lines = visible_encodable_text(message.text, encoding).split("\n")
        blocks.append(f"=== message {message.message_id} from {message.sender} to {message.target}{folder_part} ===\n"
                      + "".join(f"{MESSAGE_BODY_LINE_PREFIX} {line}\n" if line else f"{MESSAGE_BODY_LINE_PREFIX}\n" for line in body_lines)
                      + f"=== end of message {message.message_id} from {message.sender} ===\n")
    return "".join(blocks)


def print_and_record_new_messages(folder: str, reader: str, is_peek: bool) -> int:
    """Prints reader's new messages in folder and, unless is_peek, records them as read after printing. Returns how many it printed."""
    last_read_ids = load_last_read_ids(folder, reader)
    messages, updated_last_read_ids = read_new_messages(folder, reader, last_read_ids)
    if messages:
        sys.stdout.write(printable_messages(messages, sys.stdout.encoding or "utf-8"))
        sys.stdout.flush()
    if not is_peek and updated_last_read_ids != last_read_ids:
        save_last_read_ids(folder, reader, updated_last_read_ids)
    return len(messages)


def message_folder_paths() -> list[str]:
    """Returns the folders of BPSA_MESSAGE_FOLDERS: one per line, stripped, empty lines dropped (as BPSA_MCP)."""
    return [line.strip() for line in (get_env("BPSA_MESSAGE_FOLDERS") or "").splitlines() if line.strip()]


def default_message_folder() -> str | None:
    """Returns the folder of BPSA_MESSAGE_FOLDERS when it names exactly one; otherwise None."""
    folders = message_folder_paths()
    return folders[0] if len(folders) == 1 else None


class MessageSettingsError(ValueError):
    """BPSA_AGENT_NAME, BPSA_MESSAGE_FOLDERS or BPSA_MESSAGE_POLL_SECONDS is invalid; str() is a one-line reason."""


class MessageFolder(NamedTuple):
    label: str  # last component of path; it names the folder when an agent has several
    path: str  # absolute


class MessageSettings(NamedTuple):
    agent_name: str
    folders: tuple[MessageFolder, ...]
    poll_seconds: float


def message_folder_label(path: str) -> str:
    return os.path.basename(os.path.abspath(path))


def message_poll_seconds() -> float:
    """Returns BPSA_MESSAGE_POLL_SECONDS, DEFAULT_MESSAGE_POLL_SECONDS when unset or empty. Raises MessageSettingsError unless it is a finite number above 0."""
    text = (get_env("BPSA_MESSAGE_POLL_SECONDS") or "").strip()
    if not text:
        return DEFAULT_MESSAGE_POLL_SECONDS
    try:
        poll_seconds = float(text)
    except ValueError:
        poll_seconds = math.nan
    if not (math.isfinite(poll_seconds) and poll_seconds > 0):
        raise MessageSettingsError(f"BPSA_MESSAGE_POLL_SECONDS is {ascii(text)}: use a number of seconds above 0, such as 60 or 2.5.")
    return poll_seconds


def load_message_settings() -> MessageSettings | None:
    """Returns the validated message settings, or None when BPSA_AGENT_NAME is unset or empty (the other two variables are then not read).
    Raises MessageSettingsError for an invalid name, no folder, a folder that is not a directory, two folders with one label, or bad poll seconds."""
    agent_name = get_env("BPSA_AGENT_NAME") or ""
    if not agent_name:
        return None
    try:
        check_agent_name(agent_name)
    except ValueError as error:
        raise MessageSettingsError(f"BPSA_AGENT_NAME: {error}") from None
    paths = message_folder_paths()
    if not paths:
        raise MessageSettingsError("BPSA_AGENT_NAME is set but BPSA_MESSAGE_FOLDERS names no folder (one folder path per line).")
    folders = []
    for path in paths:
        if not os.path.isdir(path):
            raise MessageSettingsError(f"BPSA_MESSAGE_FOLDERS: the folder {path} does not exist or is not a folder.")
        folder = MessageFolder(message_folder_label(path), os.path.abspath(path))
        if not folder.label:
            raise MessageSettingsError(f"BPSA_MESSAGE_FOLDERS: the folder {path} has no last path component to use as its label.")
        same_label_paths = [other.path for other in folders if other.label == folder.label]
        if same_label_paths:
            raise MessageSettingsError(f"BPSA_MESSAGE_FOLDERS: the folders {same_label_paths[0]} and {folder.path} have the same label "
                                       f"{folder.label!r} (the last path component); rename one.")
        folders.append(folder)
    return MessageSettings(agent_name, tuple(folders), message_poll_seconds())


class QueuedMessages(NamedTuple):
    folder: MessageFolder
    messages: list[Message]
    last_read_ids: dict[str, int]  # the reader's last-read ids in folder once these messages are delivered


def model_text_of_queued_messages(queued: list[QueuedMessages]) -> str:
    """Returns queued as the one text a model receives: MESSAGES_FOR_MODEL_HEADING, then printable_messages with the folder label.
    Encoding utf-8 escapes control characters and lone surrogates and leaves every other character as it is."""
    return MESSAGES_FOR_MODEL_HEADING + "".join(printable_messages(batch.messages, "utf-8", batch.folder.label) for batch in queued)


class MessagePoller:
    """Daemon thread that checks every folder of settings at start and then every settings.poll_seconds, and queues the new messages to
    settings.agent_name. take_queued saves the last-read ids, so a process that ends before take_queued receives the queued messages again."""

    def __init__(self, settings: MessageSettings, print_line, announce_queued):
        self.settings = settings
        self.print_line = print_line  # called with one line of plain text: why a folder check or a save of the last-read ids failed
        self.announce_queued = announce_queued  # the thread calls it without arguments each time it has queued messages
        self.messages_waiting = threading.Event()  # set exactly while messages are queued; wait on it instead of calling take_queued in a loop
        self._queue_lock = threading.Lock()
        self._queued: list[QueuedMessages] = []
        self._last_queued_ids: dict[str, dict[str, int]] = {}  # per folder path: the ids up to which the thread has queued
        self._printed_failure_reasons: dict[str, str] = {}  # per folder path: the reason of the failing check; absent after a check that worked
        self._stop_requested = threading.Event()
        self._thread = None

    def start(self):
        self._thread = threading.Thread(target=self._poll_until_stopped, name="bpsa-messages", daemon=True)
        self._thread.start()

    def stop(self):
        """Ends the thread without waiting for the poll interval; a check in progress gets MESSAGE_POLLER_STOP_SECONDS to finish."""
        self._stop_requested.set()
        if self._thread is not None:
            self._thread.join(timeout=MESSAGE_POLLER_STOP_SECONDS)
            self._thread = None

    def take_queued(self) -> list[QueuedMessages]:
        """Removes and returns the queued messages in the order queued, and saves each folder's last-read ids. When a save fails it prints one
        line and still returns the messages: this process delivers them once, a restarted one again."""
        with self._queue_lock:
            queued, self._queued = self._queued, []
            self.messages_waiting.clear()
        for batch in {batch.folder: batch for batch in queued}.values():  # the last batch of a folder holds its newest ids
            try:
                save_last_read_ids(batch.folder.path, self.settings.agent_name, batch.last_read_ids)
            except Exception as error:
                self._print_line_or_nothing(f"Message folder {batch.folder.label}: could not record the delivered messages as read "
                                            f"({str(error) or type(error).__name__}); a restarted agent receives them again.")
        return queued

    def queued_messages_line(self) -> str | None:
        """Returns one line that names every queued message and its folder, or None when none wait."""
        with self._queue_lock:
            names = [f"message {message.message_id} from {message.sender} to {message.target} in folder {batch.folder.label}"
                     for batch in self._queued for message in batch.messages]
        return "Queued for the agent: " + ", ".join(names) if names else None

    def _print_line_or_nothing(self, line: str):
        with suppress(Exception):
            self.print_line(line)

    def _poll_until_stopped(self):
        while not self._stop_requested.is_set():
            for folder in self.settings.folders:
                try:
                    if self._queue_new_messages(folder):
                        with suppress(Exception):
                            self.announce_queued()
                    self._printed_failure_reasons.pop(folder.path, None)
                except Exception as error:
                    reason = str(error) or type(error).__name__
                    if self._printed_failure_reasons.get(folder.path) != reason:
                        self._printed_failure_reasons[folder.path] = reason
                        self._print_line_or_nothing(f"Message folder {folder.label} could not be checked: {reason}")
            terminal_display_escapes()  # its first call builds the escape table, here rather than when a run formats its first message
            self._stop_requested.wait(self.settings.poll_seconds)

    def _queue_new_messages(self, folder: MessageFolder) -> bool:
        """Queues the new messages in folder and returns whether there were any. Raises what the folder check raises."""
        reader = self.settings.agent_name
        last_queued_ids = self._last_queued_ids.get(folder.path)
        if last_queued_ids is None:
            last_queued_ids = load_last_read_ids(folder.path, reader)  # kept only after a listing worked: a missing folder reads as no ids
        messages, last_read_ids = read_new_messages(folder.path, reader, last_queued_ids)
        self._last_queued_ids[folder.path] = last_read_ids
        if not messages:
            return False
        with self._queue_lock:
            self._queued.append(QueuedMessages(folder, messages, last_read_ids))
            self.messages_waiting.set()
        return True


def message_command_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--folder", help="message folder; default: BPSA_MESSAGE_FOLDERS when it names exactly one folder")
    common.add_argument("--name", help="your agent name; default: BPSA_AGENT_NAME")
    parser = argparse.ArgumentParser(prog="python -m smolagents.bp_messages", description="Send and read agent-to-agent messages in a message folder.")
    commands = parser.add_subparsers(dest="command", required=True)
    send = commands.add_parser("send", parents=[common], help="write one message and print its file name")
    send.add_argument("--to", required=True, help="an agent name or 'all'")
    send.add_argument("text", nargs="?", default="-", help="message text; '-' or absent reads it from stdin")
    new = commands.add_parser("new", parents=[common], help="print the messages you have not read and record them as read")
    wait = commands.add_parser("wait", parents=[common], help="as new, but first wait until at least one message arrives")
    wait.add_argument("--timeout", type=float, default=DEFAULT_WAIT_TIMEOUT_SECONDS,
                      help=f"seconds to wait (default {DEFAULT_WAIT_TIMEOUT_SECONDS:g}); on timeout exit code {EXIT_WAIT_TIMEOUT} with nothing printed")
    for reading_command in (new, wait):
        reading_command.add_argument("--peek", action="store_true", help="print without recording the messages as read")
    return parser


def run_message_command(args: argparse.Namespace) -> int:
    folder = args.folder if args.folder is not None else default_message_folder()
    if folder is None:
        raise ValueError("--folder is required when BPSA_MESSAGE_FOLDERS does not name exactly one folder.")
    name = args.name if args.name is not None else get_env("BPSA_AGENT_NAME")
    if name is None:
        raise ValueError("--name is required when BPSA_AGENT_NAME is unset.")
    check_agent_name(name)
    if not os.path.isdir(folder):
        raise ValueError(f"The message folder {folder} does not exist or is not a folder.")
    if args.command == "send":
        check_message_target(args.to)
        text = sys.stdin.buffer.read().decode("utf-8", "surrogateescape") if args.text == "-" else args.text
        print(write_message(folder, name, args.to, text))
        return 0
    if args.command == "new":
        print_and_record_new_messages(folder, name, args.peek)
        return 0
    if not args.timeout >= 0:
        raise ValueError("--timeout must be a number of seconds, 0 or more.")
    terminal_display_escapes()  # builds the escape table (about 1 s) before the first message, not after it
    deadline = time.monotonic() + args.timeout
    while print_and_record_new_messages(folder, name, args.peek) == 0:
        remaining_seconds = deadline - time.monotonic()
        if remaining_seconds <= 0:
            return EXIT_WAIT_TIMEOUT
        time.sleep(min(WAIT_POLL_SECONDS, remaining_seconds))
    return 0


def main(argv: list[str] | None = None) -> int:
    """Runs the send, new and wait commands. An error prints one line on stderr and returns EXIT_ERROR; Ctrl-C returns EXIT_INTERRUPTED."""
    args = message_command_parser().parse_args(argv)
    try:
        return run_message_command(args)
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return EXIT_INTERRUPTED
    except (ValueError, OSError) as error:
        print(f"Error: {stderr_literal(str(error))}", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
