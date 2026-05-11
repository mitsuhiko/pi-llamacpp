import type { ExtensionAPI, ExtensionContext } from "@earendil-works/pi-coding-agent";
import { streamOpenAICompletions } from "@earendil-works/pi-ai";
import type { AssistantMessage, Context, Model, SimpleStreamOptions } from "@earendil-works/pi-ai";
import { spawn } from "node:child_process";
import type { ChildProcess } from "node:child_process";
import { closeSync, constants, openSync, writeSync } from "node:fs";
import { createServer } from "node:net";
import {
	access,
	appendFile,
	chmod,
	mkdir,
	open as openFile,
	readdir,
	readFile,
	realpath,
	rename,
	rm,
	stat,
	unlink,
	writeFile,
} from "node:fs/promises";
import { cpus, homedir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const EXTENSION_DIR = dirname(fileURLToPath(import.meta.url));

const PROVIDER_ID = "llamacpp";
const MANAGED_BY = "pi-llamacpp-provider";

const LLAMACPP_DIR = join(homedir(), ".pi", "llamacpp");
const RUNTIME_DIR = join(LLAMACPP_DIR, "runtime");
const DOWNLOAD_DIR = join(LLAMACPP_DIR, "downloads");
const MODEL_DIR = join(LLAMACPP_DIR, "models");
const CLIENT_DIR = join(LLAMACPP_DIR, "clients");
const LOCK_DIR = join(LLAMACPP_DIR, "lock");
const STATE_FILE = join(LLAMACPP_DIR, "server.json");
const LOG_FILE = join(LLAMACPP_DIR, "log");
const LEASE_FILE = join(CLIENT_DIR, `${process.pid}.json`);

const LLAMACPP_RELEASE_TAG = process.env.LLAMACPP_RELEASE_TAG ?? "b9090";
const LLAMACPP_RELEASE_REPO = process.env.LLAMACPP_RELEASE_REPO ?? "ggml-org/llama.cpp";
// The havenoammo MTP GGUFs need llama.cpp's MTP/NextN support from PR #22673.
// The b9090 release is kept as an opt-in fallback, but the managed default builds
// a pinned PR snapshot from GitHub so these models actually load.
const LLAMACPP_RUNTIME_KIND = (process.env.LLAMACPP_RUNTIME_KIND ?? "source").toLowerCase();
const LLAMACPP_SOURCE_REF = process.env.LLAMACPP_SOURCE_REF ?? "5d5f1b46e4f56885801c86363d4677a5f72f83af";
const LLAMACPP_SOURCE_REPO = process.env.LLAMACPP_SOURCE_REPO ?? LLAMACPP_RELEASE_REPO;
const LLAMACPP_HOST = process.env.LLAMACPP_HOST ?? "127.0.0.1";
// If unset, bind llama-server to a randomly selected localhost port and publish
// that port through server.json.  Set LLAMACPP_PORT only when you explicitly want
// a stable port for debugging.
const LLAMACPP_PORT = parseOptionalPort(process.env.LLAMACPP_PORT);
const PROVIDER_BASE_URL = LLAMACPP_PORT ? apiBaseUrlForPort(LLAMACPP_PORT) : apiBaseUrlForPort(0);
const API_KEY = process.env.LLAMACPP_API_KEY ?? "llamacpp-local";

const QWEN_35B_A3B_REPO = process.env.LLAMACPP_QWEN_35B_A3B_REPO ?? process.env.LLAMACPP_QWEN_REPO ?? "havenoammo/Qwen3.6-35B-A3B-MTP-GGUF";
const QWEN_35B_A3B_REVISION =
	process.env.LLAMACPP_QWEN_35B_A3B_REVISION ??
	process.env.LLAMACPP_QWEN_REVISION ??
	"44ce525026e7e7d0e0915dc1bf83a783c813e75a";
const QWEN_27B_REPO = process.env.LLAMACPP_QWEN_27B_REPO ?? "froggeric/Qwen3.6-27B-MTP-GGUF";
const QWEN_27B_REVISION = process.env.LLAMACPP_QWEN_27B_REVISION ?? "431204640c8511573e61a7964a12cc452114a223";
const DEFAULT_CTX_SIZE = Number(process.env.LLAMACPP_CTX_SIZE ?? 262144);
const DEFAULT_MAX_TOKENS = Number(process.env.LLAMACPP_MAX_TOKENS ?? 65536);

// Qwen3.6's model card recommends different sampling defaults for thinking and
// non-thinking modes. This extension is primarily used for coding-agent work, so
// use the precise-coding thinking preset when reasoning is enabled.
const QWEN_THINKING_SAMPLING = {
	temperature: 0.6,
	top_p: 0.95,
	top_k: 20,
	min_p: 0.0,
	presence_penalty: 0.0,
	repeat_penalty: 1.0,
};
const QWEN_INSTRUCT_SAMPLING = {
	temperature: 0.7,
	top_p: 0.8,
	top_k: 20,
	min_p: 0.0,
	presence_penalty: 1.5,
	repeat_penalty: 1.0,
};

const HEARTBEAT_MS = 10_000;
const LEASE_TTL_MS = 45_000;
const LOCK_STALE_MS = 60_000;
const LOCK_TIMEOUT_MS = 30_000;
const STARTUP_LOCK_TIMEOUT_MS = 24 * 60 * 60_000;
const READY_TIMEOUT_MS = Number(process.env.LLAMACPP_READY_TIMEOUT_MS ?? 10 * 60_000);
const HTTP_CHECK_TIMEOUT_MS = 1_500;
const SHUTDOWN_GRACE_MS = 60_000;
const LOG_TAIL_BYTES = 256 * 1024;
const LOG_MAX_LINES = 2_000;
const LOG_POLL_MS = 1_000;
const WATCHDOG_POLL_MS = 2_000;
const PROGRESS_NOTIFY_MS = 750;
const PROGRESS_MAX_CHARS = 180;

const WATCHDOG_SCRIPT_NAME = "llamacpp-watchdog.sh";
const WATCHDOG_SCRIPT = process.env.LLAMACPP_WATCHDOG_SCRIPT
	? resolve(process.env.LLAMACPP_WATCHDOG_SCRIPT)
	: join(EXTENSION_DIR, WATCHDOG_SCRIPT_NAME);

type ModelQuant = "q2" | "q4" | "q8";

type ManagedModel = {
	id: string;
	name: string;
	repo: string;
	revision: string;
	quant: ModelQuant;
	bits: number;
	filename: string;
	size: number;
	sha256: string;
};

const MODELS: ManagedModel[] = [
	{
		id: "qwen-3.6-moe-2bit",
		name: "qwen-3.6-moe-2bit",
		repo: QWEN_35B_A3B_REPO,
		revision: QWEN_35B_A3B_REVISION,
		quant: "q2",
		bits: 2,
		filename: "Qwen3.6-35B-A3B-MTP-UD-Q2_K_XL.gguf",
		size: 13_188_092_320,
		sha256: "c723e9516bf0b21a85390277913a1642f4644b221309518daac3a8d51a3a5850",
	},
	{
		id: "qwen-3.6-moe-4bit",
		name: "qwen-3.6-moe-4bit",
		repo: QWEN_35B_A3B_REPO,
		revision: QWEN_35B_A3B_REVISION,
		quant: "q4",
		bits: 4,
		filename: "Qwen3.6-35B-A3B-MTP-UD-Q4_K_XL.gguf",
		size: 23_257_919_904,
		sha256: "ab94e2da12d2bdc22777ba1b7422bbf8d5d9d0bee1164ca7343a0cee3310038a",
	},
	{
		id: "qwen-3.6-moe-8bit",
		name: "qwen-3.6-moe-8bit",
		repo: QWEN_35B_A3B_REPO,
		revision: QWEN_35B_A3B_REVISION,
		quant: "q8",
		bits: 8,
		filename: "Qwen3.6-35B-A3B-MTP-UD-Q8_K_XL.gguf",
		size: 39_348_646_304,
		sha256: "3720209c5729265b0967445e3f4d2d46d6455bc21123958fd4cac203f3277478",
	},
	{
		id: "qwen-3.6-dense-2bit",
		name: "qwen-3.6-dense-2bit",
		repo: QWEN_27B_REPO,
		revision: QWEN_27B_REVISION,
		quant: "q2",
		bits: 2,
		filename: "Qwen3.6-27B-IQ2_M-mtp.gguf",
		size: 10_455_916_448,
		sha256: "69c06c105d5e7d8a5ad9c0ead59fbcfdf7e2a7ea7d9338aadfe70fc9b0f133bf",
	},
	{
		id: "qwen-3.6-dense-4bit",
		name: "qwen-3.6-dense-4bit",
		repo: QWEN_27B_REPO,
		revision: QWEN_27B_REVISION,
		quant: "q4",
		bits: 4,
		filename: "Qwen3.6-27B-Q4_K_M-mtp.gguf",
		size: 16_998_723_232,
		sha256: "c2275978182b91ec0f0a2e334e37e4fbfc8385eb9b3cdb6d5d4f7e23fce3b557",
	},
	{
		id: "qwen-3.6-dense-8bit",
		name: "qwen-3.6-dense-8bit",
		repo: QWEN_27B_REPO,
		revision: QWEN_27B_REVISION,
		quant: "q8",
		bits: 8,
		filename: "Qwen3.6-27B-Q8_0-mtp.gguf",
		size: 29_047_086_752,
		sha256: "15de87dd41f9a05c2b8938c4a7234280a5b148f2ac047b7f80abca548a768b2f",
	},
];

const MODEL_BY_ID = new Map(MODELS.map((model) => [model.id, model]));

type RuntimeAsset = {
	name: string;
	url: string;
	extension: "tar.gz";
};

type ServerState = {
	managedBy: string;
	pid: number;
	baseUrl: string;
	port: number;
	cwd: string;
	binary: string;
	args: string[];
	modelId: string;
	modelName: string;
	quant: ModelQuant;
	modelFile: string;
	releaseTag: string;
	assetName?: string;
	startedAt: number;
	startedAtIso: string;
	stopping?: boolean;
	stoppingAt?: number;
	stoppingAtIso?: string;
};

type Lease = {
	managedBy: string;
	usesLlamaCpp: true;
	pid: number;
	processStart: string;
	cwd: string;
	modelId: string;
	quant: ModelQuant;
	startedAt: number;
	updatedAt: number;
	updatedAtIso: string;
};

type StatusCallback = (message: string | undefined) => void;
type RunLoggedOptions = { onStatus?: StatusCallback; progressPrefix?: string };

type LogTui = { terminal: { rows: number }; requestRender: (force?: boolean) => void };
type LogTheme = { fg: (color: string, text: string) => string };
type Component = { render(width: number): string[]; handleInput?(data: string): void; invalidate(): void };

let heartbeat: ReturnType<typeof setInterval> | undefined;
let startupPromise: Promise<void> | undefined;
let activeProviderContext: ExtensionContext | undefined;
let activeSetupChild: ChildProcess | undefined;
let resolvedRuntimeDir: string | undefined;
let resolvedRuntimeAsset: RuntimeAsset | undefined;
let activeLeaseModel: ManagedModel | undefined;
let leaseStartedAt = Date.now();
let ownProcessStart: string | undefined;
let leaseActive = false;
let watchdogStarted = false;
let runtimeDisposed = false;
let shuttingDown = false;
let writeSeq = 0;

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

function parseOptionalPort(value: string | undefined): number | undefined {
	if (!value) return undefined;
	const port = Number(value);
	if (!Number.isInteger(port) || port <= 0 || port > 65535) {
		throw new Error(`Invalid LLAMACPP_PORT=${value}; expected 1-65535`);
	}
	return port;
}

function baseUrlForPort(port: number): string {
	return `http://${LLAMACPP_HOST}:${port}`;
}

function apiBaseUrlForPort(port: number): string {
	return `${baseUrlForPort(port)}/v1`;
}

async function chooseServerPort(): Promise<number> {
	if (LLAMACPP_PORT) return LLAMACPP_PORT;
	return new Promise((resolvePromise, reject) => {
		const server = createServer();
		server.unref();
		server.on("error", reject);
		server.listen(0, LLAMACPP_HOST, () => {
			const address = server.address();
			const port = typeof address === "object" && address ? address.port : undefined;
			server.close((error) => {
				if (error) reject(error);
				else if (port) resolvePromise(port);
				else reject(new Error("Could not allocate a random llama-server port"));
			});
		});
	});
}

function describeApiBaseUrl(): string {
	return LLAMACPP_PORT ? apiBaseUrlForPort(LLAMACPP_PORT) : "dynamic localhost port";
}

function describeError(error: unknown): string {
	return error instanceof Error ? error.message : String(error);
}

function isLockTimeout(error: unknown): boolean {
	return describeError(error).includes("Timed out waiting for llama.cpp lifecycle lock");
}

function isPidAlive(pid: unknown): pid is number {
	if (typeof pid !== "number" || !Number.isFinite(pid) || pid <= 0) return false;
	try {
		process.kill(pid, 0);
		return true;
	} catch (error: any) {
		return error?.code === "EPERM";
	}
}

function shellQuote(value: string): string {
	return /^[A-Za-z0-9_./:=+@,?-]+$/.test(value) ? value : `'${value.replace(/'/g, `'\\''`)}'`;
}

function isKey(data: string, key: "escape" | "up" | "down" | "home" | "end" | "pageUp" | "pageDown"): boolean {
	switch (key) {
		case "escape":
			return data === "\x1b";
		case "up":
			return data === "\x1b[A" || data === "\x1bOA";
		case "down":
			return data === "\x1b[B" || data === "\x1bOB";
		case "home":
			return data === "\x1b[H" || data === "\x1bOH" || data === "\x1b[1~";
		case "end":
			return data === "\x1b[F" || data === "\x1bOF" || data === "\x1b[4~";
		case "pageUp":
			return data === "\x1b[5~";
		case "pageDown":
			return data === "\x1b[6~";
	}
}

const ANSI_RE = /\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07]*(?:\x07|\x1b\\)|_[^\x07]*(?:\x07|\x1b\\))/g;

function stripAnsi(value: string): string {
	return value.replace(ANSI_RE, "");
}

function truncateText(value: string, width: number, ellipsis = "", pad = false): string {
	if (width <= 0) return "";
	let text = stripAnsi(value);
	if (text.length > width) {
		const suffix = ellipsis.length < width ? ellipsis : "";
		text = text.slice(0, width - suffix.length) + suffix;
	}
	return pad ? text + " ".repeat(Math.max(0, width - text.length)) : text;
}

function formatBytes(bytes: number): string {
	const units = ["B", "KB", "MB", "GB", "TB"];
	let value = bytes;
	let unit = 0;
	while (value >= 1024 && unit < units.length - 1) {
		value /= 1024;
		unit++;
	}
	return `${value.toFixed(unit === 0 ? 0 : 1)} ${units[unit]}`;
}

async function ensureDirs(): Promise<void> {
	await mkdir(CLIENT_DIR, { recursive: true });
	await mkdir(DOWNLOAD_DIR, { recursive: true });
	await mkdir(MODEL_DIR, { recursive: true });
}

async function readJson<T>(file: string): Promise<T | undefined> {
	try {
		return JSON.parse(await readFile(file, "utf8")) as T;
	} catch {
		return undefined;
	}
}

async function writeJsonAtomic(file: string, data: unknown): Promise<void> {
	await mkdir(dirname(file), { recursive: true });
	const tmp = `${file}.${process.pid}.${Date.now()}.${++writeSeq}.tmp`;
	await writeFile(tmp, `${JSON.stringify(data, null, 2)}\n`, "utf8");
	await rename(tmp, file);
}

async function removeFile(file: string): Promise<void> {
	try {
		await unlink(file);
	} catch (error: any) {
		if (error?.code !== "ENOENT") throw error;
	}
}

async function appendLog(text: string): Promise<void> {
	await mkdir(LLAMACPP_DIR, { recursive: true });
	await appendFile(LOG_FILE, text, "utf8");
}

async function readLogTail(): Promise<string[]> {
	try {
		const info = await stat(LOG_FILE);
		if (!info.isFile()) return [`${LOG_FILE} exists but is not a file`];

		const bytes = Math.min(info.size, LOG_TAIL_BYTES);
		const buffer = Buffer.alloc(bytes);
		const file = await openFile(LOG_FILE, "r");
		try {
			await file.read(buffer, 0, bytes, info.size - bytes);
		} finally {
			await file.close();
		}

		let text = stripAnsi(buffer.toString("utf8")).replace(/\r\n/g, "\n").replace(/\r/g, "\n");
		if (info.size > bytes) {
			const firstNewline = text.indexOf("\n");
			if (firstNewline >= 0) text = text.slice(firstNewline + 1);
			text = `[showing last ${formatBytes(bytes)} of ${formatBytes(info.size)} from ${LOG_FILE}]\n${text}`;
		}

		const lines = text.split("\n");
		if (lines.at(-1) === "") lines.pop();
		return lines.slice(-LOG_MAX_LINES);
	} catch (error: any) {
		if (error?.code === "ENOENT") return [`No llama.cpp log yet: ${LOG_FILE}`];
		return [`Failed to read ${LOG_FILE}: ${describeError(error)}`];
	}
}

class LlamaCppLogViewer implements Component {
	private lines: string[] = [];
	private scrollFromBottom = 0;
	private timer: ReturnType<typeof setInterval> | undefined;
	private version = 0;
	private cachedWidth = 0;
	private cachedRows = 0;
	private cachedVersion = -1;
	private cachedScroll = -1;
	private cachedLines: string[] = [];

	constructor(
		private tui: LogTui,
		private theme: LogTheme,
		private done: () => void,
	) {
		void this.refresh();
		this.timer = setInterval(() => void this.refresh(), LOG_POLL_MS);
		this.timer.unref?.();
	}

	private async refresh(): Promise<void> {
		const wasFollowing = this.scrollFromBottom === 0;
		this.lines = await readLogTail();
		this.version++;
		if (wasFollowing) this.scrollFromBottom = 0;
		this.invalidate();
		this.tui.requestRender();
	}

	private viewportHeight(): number {
		return Math.max(8, Math.min(40, this.tui.terminal.rows - 6));
	}

	private bodyHeight(): number {
		return Math.max(1, this.viewportHeight() - 4);
	}

	private clampScroll(): void {
		this.scrollFromBottom = Math.max(0, Math.min(this.scrollFromBottom, Math.max(0, this.lines.length - this.bodyHeight())));
	}

	handleInput(data: string): void {
		const page = Math.max(1, this.bodyHeight() - 2);
		if (isKey(data, "escape") || data === "q") {
			this.done();
			return;
		}
		if (isKey(data, "up") || data === "k") this.scrollFromBottom++;
		else if (isKey(data, "down") || data === "j") this.scrollFromBottom--;
		else if (isKey(data, "home")) this.scrollFromBottom = this.lines.length;
		else if (isKey(data, "end")) this.scrollFromBottom = 0;
		else if (isKey(data, "pageUp") || data === "b") this.scrollFromBottom += page;
		else if (isKey(data, "pageDown") || data === "f") this.scrollFromBottom -= page;
		else return;

		this.clampScroll();
		this.invalidate();
		this.tui.requestRender();
	}

	private borderLine(left: string, fill: string, right: string, width: number, title?: string): string {
		const innerWidth = Math.max(0, width - 2);
		let inner = this.theme.fg("border", fill.repeat(innerWidth));
		if (title) {
			const rawTitle = truncateText(` ${title} `, innerWidth);
			const fillWidth = Math.max(0, innerWidth - rawTitle.length);
			inner = this.theme.fg("accent", rawTitle) + this.theme.fg("border", fill.repeat(fillWidth));
		}
		return this.theme.fg("border", left) + inner + this.theme.fg("border", right);
	}

	private row(text: string, width: number, color?: (value: string) => string): string {
		const innerWidth = Math.max(0, width - 4);
		const content = truncateText(text.replace(/\t/g, "   "), innerWidth, "", true);
		return this.theme.fg("border", "│") + " " + (color ? color(content) : content) + " " + this.theme.fg("border", "│");
	}

	render(width: number): string[] {
		const height = this.viewportHeight();
		if (
			this.cachedWidth === width &&
			this.cachedRows === height &&
			this.cachedVersion === this.version &&
			this.cachedScroll === this.scrollFromBottom
		) {
			return this.cachedLines;
		}

		this.clampScroll();
		const bodyHeight = this.bodyHeight();
		const start = Math.max(0, this.lines.length - bodyHeight - this.scrollFromBottom);
		const visible = this.lines.slice(start, start + bodyHeight);
		while (visible.length < bodyHeight) visible.unshift("");

		const state = this.scrollFromBottom === 0 ? "live" : `${this.scrollFromBottom} lines up`;
		const title = `llama.cpp log • ${state}`;
		const help = `↑↓ scroll • Pg page • End live • q/Esc close • ${LOG_FILE}`;
		const lines = [
			this.borderLine("╭", "─", "╮", width, title),
			...visible.map((line) => this.row(line, width)),
			this.row(help, width, (value) => this.theme.fg("dim", value)),
			this.borderLine("╰", "─", "╯", width),
		];

		this.cachedWidth = width;
		this.cachedRows = height;
		this.cachedVersion = this.version;
		this.cachedScroll = this.scrollFromBottom;
		this.cachedLines = lines;
		return this.cachedLines;
	}

	invalidate(): void {
		this.cachedWidth = 0;
	}

	dispose(): void {
		if (this.timer) {
			clearInterval(this.timer);
			this.timer = undefined;
		}
	}
}

async function execCapture(command: string, args: string[], timeoutMs = 2_000): Promise<string | undefined> {
	return new Promise((resolvePromise) => {
		let stdout = "";
		let stderr = "";
		let settled = false;
		let child: ChildProcess | undefined;

		const timeout = setTimeout(() => {
			try {
				child?.kill("SIGTERM");
			} catch {}
			finish(undefined);
		}, timeoutMs);
		timeout.unref?.();

		const finish = (value: string | undefined) => {
			if (settled) return;
			settled = true;
			clearTimeout(timeout);
			resolvePromise(value);
		};

		try {
			child = spawn(command, args, { stdio: ["ignore", "pipe", "pipe"] });
		} catch {
			finish(undefined);
			return;
		}

		child.stdout?.setEncoding("utf8");
		child.stderr?.setEncoding("utf8");
		child.stdout?.on("data", (chunk) => (stdout += chunk));
		child.stderr?.on("data", (chunk) => (stderr += chunk));
		child.on("error", () => finish(undefined));
		child.on("close", (code) => finish(code === 0 ? stdout : stdout || stderr || undefined));
	});
}

async function processArgs(pid: number): Promise<string | undefined> {
	return (await execCapture("ps", ["-p", String(pid), "-o", "args="], 2_000))?.trim();
}

async function processStart(pid: number): Promise<string | undefined> {
	return (await execCapture("ps", ["-p", String(pid), "-o", "lstart="], 2_000))?.trim() || undefined;
}

async function getOwnProcessStart(): Promise<string> {
	ownProcessStart ??= (await processStart(process.pid)) ?? "unknown";
	return ownProcessStart;
}

async function isLeaseForLiveProcess(lease: Lease | undefined): Promise<boolean> {
	if (!lease || lease.managedBy !== MANAGED_BY || lease.usesLlamaCpp !== true) return false;
	if (!isPidAlive(lease.pid)) return false;
	if (!lease.processStart) return false;
	const currentStart = await processStart(lease.pid);
	return currentStart === lease.processStart;
}

async function looksLikeLlamaServer(pid: number): Promise<boolean> {
	const args = await processArgs(pid);
	return !!args && /(^|[/\s])llama-server(?:\.exe)?(\s|$)/.test(args);
}

async function findListeningPid(port: number | undefined): Promise<number | undefined> {
	if (!port) return undefined;
	const output = await execCapture("lsof", ["-nP", `-tiTCP:${port}`, "-sTCP:LISTEN"], 2_000);
	for (const line of (output ?? "").split(/\r?\n/)) {
		const pid = Number(line.trim());
		if (Number.isInteger(pid) && isPidAlive(pid)) return pid;
	}
	return undefined;
}

async function findListeningLlamaServerPid(port: number | undefined): Promise<number | undefined> {
	const pid = await findListeningPid(port);
	if (pid && (await looksLikeLlamaServer(pid))) return pid;
	return undefined;
}

async function describeProcess(pid: number): Promise<string> {
	return (await processArgs(pid)) || `pid ${pid}`;
}

async function resolveWatchdogScript(): Promise<string> {
	try {
		await access(WATCHDOG_SCRIPT, constants.F_OK);
		return WATCHDOG_SCRIPT;
	} catch {
		throw new Error(`Cannot find bundled ${WATCHDOG_SCRIPT_NAME} at ${WATCHDOG_SCRIPT}`);
	}
}

async function cleanupOldWatchdogs(): Promise<void> {
	const output = await execCapture("ps", ["axww", "-o", "pid=,args="], 2_000);
	const invocation = `${WATCHDOG_SCRIPT_NAME} ${LLAMACPP_DIR}`;
	for (const line of (output ?? "").split(/\r?\n/)) {
		const match = line.trim().match(/^(\d+)\s+(.*)$/);
		if (!match) continue;
		const pid = Number(match[1]);
		const args = match[2] ?? "";
		if (pid === process.pid || !args.includes(invocation)) continue;
		if (isPidAlive(pid)) return;
	}
}

async function hasRunningWatchdog(): Promise<boolean> {
	const output = await execCapture("ps", ["axww", "-o", "pid=,args="], 2_000);
	const invocation = `${WATCHDOG_SCRIPT_NAME} ${LLAMACPP_DIR}`;
	for (const line of (output ?? "").split(/\r?\n/)) {
		const match = line.trim().match(/^(\d+)\s+(.*)$/);
		if (!match) continue;
		const pid = Number(match[1]);
		const args = match[2] ?? "";
		if (pid !== process.pid && args.includes(invocation)) return true;
	}
	return false;
}

async function ensureWatchdog(): Promise<void> {
	if (watchdogStarted) return;
	await mkdir(LLAMACPP_DIR, { recursive: true });
	await cleanupOldWatchdogs();
	const watchdogScript = await resolveWatchdogScript();

	if (await hasRunningWatchdog()) {
		watchdogStarted = true;
		return;
	}

	const logFd = openSync(LOG_FILE, "a");
	try {
		const child = spawn("/bin/sh", [watchdogScript, LLAMACPP_DIR], {
			detached: true,
			stdio: ["ignore", logFd, logFd],
			env: {
				...process.env,
				LLAMACPP_DIR,
				LLAMACPP_CLIENT_DIR: CLIENT_DIR,
				LLAMACPP_STATE_FILE: STATE_FILE,
				LLAMACPP_LOG_FILE: LOG_FILE,
				LLAMACPP_PORT: LLAMACPP_PORT ? String(LLAMACPP_PORT) : "",
				LLAMACPP_LEASE_TTL_S: String(Math.ceil(LEASE_TTL_MS / 1000)),
				LLAMACPP_WATCHDOG_POLL_S: String(Math.max(1, Math.ceil(WATCHDOG_POLL_MS / 1000))),
				LLAMACPP_SHUTDOWN_GRACE_S: String(Math.ceil(SHUTDOWN_GRACE_MS / 1000)),
			},
		});
		child.unref();
		watchdogStarted = true;
	} finally {
		closeSync(logFd);
	}
}

function formatCurlProgress(line: string): string | undefined {
	const fields = line.trim().split(/\s+/);
	if (fields.length < 12) return undefined;
	if (!/^\d+(?:\.\d+)?$/.test(fields[0]) || !/^\d+(?:\.\d+)?$/.test(fields[2])) return undefined;

	const total = fields[1];
	const percent = fields[2];
	const received = fields[3];
	const left = fields[10];
	const speed = fields[11];
	if (!total || !received) return undefined;

	const details = [`${percent}%`];
	if (speed && speed !== "0") details.push(`${speed}/s`);
	if (left && left !== "--:--:--") details.push(`${left} left`);
	return `${received} / ${total} (${details.join(", ")})`;
}

function compactProgressLine(rawLine: string): string | undefined {
	let line = stripAnsi(rawLine)
		.replace(/[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]/g, "")
		.replace(/\s+/g, " ")
		.trim();
	if (!line) return undefined;
	if (/^% Total\b/.test(line) || /^Dload\s+Upload\b/.test(line)) return undefined;

	line = formatCurlProgress(line) ?? line;
	if (line.length > PROGRESS_MAX_CHARS) line = `${line.slice(0, PROGRESS_MAX_CHARS - 1)}…`;
	return line;
}

function createProgressReporter(prefix: string, onStatus?: StatusCallback) {
	let lineBuffer = "";
	let latest: string | undefined;
	let emitted: string | undefined;
	let lastEmit = 0;

	const maybeEmit = (force = false) => {
		if (!onStatus || !latest || latest === emitted) return;
		const now = Date.now();
		if (!force && now - lastEmit < PROGRESS_NOTIFY_MS) return;
		emitted = latest;
		lastEmit = now;
		onStatus(`${prefix}: ${latest}`);
	};

	const processLine = (line: string) => {
		const progress = compactProgressLine(line);
		if (!progress) return;
		latest = progress;
		maybeEmit(false);
	};

	const onChunk = (chunk: Buffer | string) => {
		const text = chunk.toString();
		let start = 0;
		for (let i = 0; i < text.length; i++) {
			const ch = text[i];
			if (ch !== "\r" && ch !== "\n") continue;
			processLine(lineBuffer + text.slice(start, i));
			lineBuffer = "";
			if (ch === "\r" && text[i + 1] === "\n") i++;
			start = i + 1;
		}
		lineBuffer += text.slice(start);
		if (lineBuffer) processLine(lineBuffer);
		if (lineBuffer.length > 4096) lineBuffer = "";
	};

	const flush = () => {
		if (lineBuffer) {
			processLine(lineBuffer);
			lineBuffer = "";
		}
		maybeEmit(true);
	};

	return { onChunk, flush };
}

async function runLogged(command: string, args: string[], cwd: string, label: string, options: RunLoggedOptions = {}): Promise<void> {
	if (runtimeDisposed || shuttingDown) throw new Error(`${label} cancelled`);

	await appendLog(`\n[${new Date().toISOString()}] ${label}\n$ ${[command, ...args].map(shellQuote).join(" ")}\n`);

	const logFd = openSync(LOG_FILE, "a");
	const progress = options.progressPrefix ? createProgressReporter(options.progressPrefix, options.onStatus) : undefined;
	let closed = false;
	const writeLogChunk = (chunk: Buffer | string) => {
		if (closed) return;
		try {
			if (typeof chunk === "string") writeSync(logFd, chunk);
			else writeSync(logFd, chunk);
		} catch {}
	};
	const closeLog = () => {
		if (!closed) {
			closed = true;
			closeSync(logFd);
		}
	};

	await new Promise<void>((resolvePromise, reject) => {
		let child: ChildProcess;
		try {
			child = spawn(command, args, {
				cwd,
				detached: process.platform !== "win32",
				stdio: ["ignore", "pipe", "pipe"],
				env: process.env,
			});
		} catch (error) {
			progress?.flush();
			closeLog();
			reject(error);
			return;
		}

		activeSetupChild = child;
		const handleOutput = (chunk: Buffer) => {
			writeLogChunk(chunk);
			progress?.onChunk(chunk);
		};
		child.stdout?.on("data", handleOutput);
		child.stderr?.on("data", handleOutput);

		const finish = (error?: Error) => {
			if (activeSetupChild === child) activeSetupChild = undefined;
			progress?.flush();
			closeLog();
			if (error) reject(error);
			else resolvePromise();
		};

		child.on("error", (error) => finish(error));
		child.on("close", (code, signal) => {
			if (runtimeDisposed || shuttingDown) {
				finish(new Error(`${label} cancelled`));
			} else if (code === 0) {
				finish();
			} else {
				finish(new Error(`${label} failed (${signal ? `signal ${signal}` : `exit ${code}`}); see ${LOG_FILE}`));
			}
		});
	});
}

function killActiveSetupChild(): void {
	const child = activeSetupChild;
	if (!child?.pid) return;
	try {
		process.kill(process.platform === "win32" ? child.pid : -child.pid, "SIGTERM");
	} catch {}
}

function selectRuntimeAsset(): RuntimeAsset {
	const overrideUrl = process.env.LLAMACPP_RELEASE_ASSET_URL;
	const overrideName = process.env.LLAMACPP_RELEASE_ASSET_NAME ?? overrideUrl?.split("/").pop();
	if (overrideUrl && overrideName) return { name: overrideName, url: overrideUrl, extension: "tar.gz" };

	let platform: string;
	let arch: string;
	if (process.platform === "darwin") platform = "macos";
	else if (process.platform === "linux") platform = "ubuntu";
	else throw new Error(`Unsupported platform for managed llama.cpp release: ${process.platform}`);

	if (process.arch === "arm64") arch = "arm64";
	else if (process.arch === "x64") arch = "x64";
	else throw new Error(`Unsupported architecture for managed llama.cpp release: ${process.arch}`);

	const name = `llama-${LLAMACPP_RELEASE_TAG}-bin-${platform}-${arch}.tar.gz`;
	return {
		name,
		url: `https://github.com/${LLAMACPP_RELEASE_REPO}/releases/download/${LLAMACPP_RELEASE_TAG}/${name}`,
		extension: "tar.gz",
	};
}

function runtimeInstallDir(asset: RuntimeAsset): string {
	const safeAsset = asset.name.replace(/[^A-Za-z0-9_.-]+/g, "_").replace(/\.tar\.gz$/, "");
	return join(RUNTIME_DIR, LLAMACPP_RELEASE_TAG, safeAsset);
}

async function findLlamaServerBinary(runtimeDir: string): Promise<string | undefined> {
	const names = process.platform === "win32" ? ["llama-server.exe"] : ["llama-server"];
	for (const name of names) {
		const candidate = join(runtimeDir, name);
		try {
			await access(candidate, constants.X_OK);
			return candidate;
		} catch {}
	}
	return undefined;
}

async function downloadFile(url: string, destination: string, label: string, onStatus?: StatusCallback): Promise<void> {
	await mkdir(dirname(destination), { recursive: true });
	const partial = `${destination}.part`;
	await runLogged("curl", ["--fail", "--location", "--continue-at", "-", "--output", partial, url], LLAMACPP_DIR, label, {
		onStatus,
		progressPrefix: label,
	});
	await rename(partial, destination);
}

async function ensureReleaseRuntime(onStatus?: StatusCallback): Promise<string> {
	const asset = selectRuntimeAsset();
	resolvedRuntimeAsset = asset;
	const targetDir = runtimeInstallDir(asset);
	const existing = await findLlamaServerBinary(targetDir);
	if (existing) return targetDir;

	onStatus?.(`downloading llama.cpp ${LLAMACPP_RELEASE_TAG}`);
	await mkdir(DOWNLOAD_DIR, { recursive: true });
	const archive = join(DOWNLOAD_DIR, asset.name);
	try {
		await access(archive, constants.F_OK);
	} catch {
		await downloadFile(asset.url, archive, `downloading ${asset.name}`, onStatus);
	}

	onStatus?.(`installing llama.cpp ${LLAMACPP_RELEASE_TAG}`);
	const tmpDir = `${targetDir}.tmp.${process.pid}.${Date.now()}`;
	await rm(tmpDir, { recursive: true, force: true });
	await mkdir(tmpDir, { recursive: true });
	try {
		await runLogged("tar", ["-xzf", archive, "-C", tmpDir, "--strip-components", "1"], LLAMACPP_DIR, `extract ${asset.name}`, {
			onStatus,
			progressPrefix: `installing ${asset.name}`,
		});
		const binary = await findLlamaServerBinary(tmpDir);
		if (!binary) throw new Error(`Extracted ${asset.name} but did not find llama-server`);
		await chmod(binary, 0o755).catch(() => {});
		await mkdir(dirname(targetDir), { recursive: true });
		await rm(targetDir, { recursive: true, force: true });
		await rename(tmpDir, targetDir);
	} catch (error) {
		await rm(tmpDir, { recursive: true, force: true }).catch(() => {});
		throw error;
	}

	return targetDir;
}

function sourceRefSafe(ref: string): string {
	return ref.replace(/[^A-Za-z0-9_.-]+/g, "_");
}

function sourceArchiveUrl(): string {
	return process.env.LLAMACPP_SOURCE_URL ?? `https://github.com/${LLAMACPP_SOURCE_REPO}/archive/${LLAMACPP_SOURCE_REF}.tar.gz`;
}

async function ensureSourceCheckout(onStatus?: StatusCallback): Promise<string> {
	const safeRef = sourceRefSafe(LLAMACPP_SOURCE_REF);
	const sourceDir = join(LLAMACPP_DIR, "source", safeRef);
	try {
		await access(join(sourceDir, "CMakeLists.txt"), constants.F_OK);
		return sourceDir;
	} catch {}

	onStatus?.(`downloading llama.cpp source ${LLAMACPP_SOURCE_REF.slice(0, 12)}`);
	await mkdir(DOWNLOAD_DIR, { recursive: true });
	const archive = join(DOWNLOAD_DIR, `llama-${safeRef}-source.tar.gz`);
	try {
		await access(archive, constants.F_OK);
	} catch {
		await downloadFile(sourceArchiveUrl(), archive, `downloading llama.cpp source ${LLAMACPP_SOURCE_REF.slice(0, 12)}`, onStatus);
	}

	onStatus?.(`installing llama.cpp source ${LLAMACPP_SOURCE_REF.slice(0, 12)}`);
	const tmpDir = `${sourceDir}.tmp.${process.pid}.${Date.now()}`;
	await rm(tmpDir, { recursive: true, force: true });
	await mkdir(tmpDir, { recursive: true });
	try {
		await runLogged("tar", ["-xzf", archive, "-C", tmpDir, "--strip-components", "1"], LLAMACPP_DIR, `extract llama.cpp source`, {
			onStatus,
			progressPrefix: "installing llama.cpp source",
		});
		await access(join(tmpDir, "CMakeLists.txt"), constants.F_OK);
		await mkdir(dirname(sourceDir), { recursive: true });
		await rm(sourceDir, { recursive: true, force: true });
		await rename(tmpDir, sourceDir);
	} catch (error) {
		await rm(tmpDir, { recursive: true, force: true }).catch(() => {});
		throw error;
	}
	return sourceDir;
}

function splitArgs(value: string | undefined): string[] {
	return value?.match(/(?:[^\s"']+|"[^"]*"|'[^']*')+/g)?.map((part) => part.replace(/^(["'])(.*)\1$/, "$2")) ?? [];
}

async function ensureSourceRuntime(onStatus?: StatusCallback): Promise<string> {
	const sourceDir = await ensureSourceCheckout(onStatus);
	const buildDir = join(sourceDir, "build");
	const runtimeDir = join(buildDir, "bin");
	const existing = await findLlamaServerBinary(runtimeDir);
	if (existing) return runtimeDir;

	onStatus?.("configuring llama.cpp");
	await runLogged(
		"cmake",
		[
			"-S",
			sourceDir,
			"-B",
			buildDir,
			"-DCMAKE_BUILD_TYPE=Release",
			"-DLLAMA_BUILD_TESTS=OFF",
			...splitArgs(process.env.LLAMACPP_CMAKE_ARGS),
		],
		LLAMACPP_DIR,
		"configure llama.cpp",
		{ onStatus, progressPrefix: "configuring llama.cpp" },
	);

	onStatus?.("building llama-server");
	const jobs = process.env.LLAMACPP_BUILD_JOBS ?? String(Math.max(2, Math.min(8, cpus().length || 2)));
	await runLogged(
		"cmake",
		["--build", buildDir, "--config", "Release", "--target", "llama-server", "-j", jobs, ...splitArgs(process.env.LLAMACPP_BUILD_ARGS)],
		LLAMACPP_DIR,
		"build llama-server",
		{ onStatus, progressPrefix: "building llama-server" },
	);

	const binary = await findLlamaServerBinary(runtimeDir);
	if (!binary) throw new Error(`Built llama.cpp but did not find llama-server in ${runtimeDir}`);
	await chmod(binary, 0o755).catch(() => {});
	return runtimeDir;
}

async function ensureRuntime(onStatus?: StatusCallback): Promise<string> {
	if (resolvedRuntimeDir) return resolvedRuntimeDir;

	const forcedBinary = process.env.LLAMACPP_SERVER_BINARY;
	if (forcedBinary) {
		const binary = resolve(forcedBinary);
		await access(binary, constants.X_OK);
		resolvedRuntimeDir = dirname(binary);
		return resolvedRuntimeDir;
	}

	const forcedRuntime = process.env.LLAMACPP_RUNTIME_DIR;
	if (forcedRuntime) {
		const dir = resolve(forcedRuntime);
		const binary = await findLlamaServerBinary(dir);
		if (!binary) throw new Error(`LLAMACPP_RUNTIME_DIR=${dir} does not contain an executable llama-server`);
		resolvedRuntimeDir = dir;
		return dir;
	}

	if (LLAMACPP_RUNTIME_KIND === "release") {
		resolvedRuntimeDir = await ensureReleaseRuntime(onStatus);
	} else if (LLAMACPP_RUNTIME_KIND === "source") {
		resolvedRuntimeDir = await ensureSourceRuntime(onStatus);
	} else {
		throw new Error(`Invalid LLAMACPP_RUNTIME_KIND=${LLAMACPP_RUNTIME_KIND}; expected source or release`);
	}
	return resolvedRuntimeDir;
}

function modelCachePath(model: ManagedModel): string {
	return join(MODEL_DIR, ...model.repo.split("/"), model.filename);
}

function huggingFaceModelUrl(model: ManagedModel): string {
	return `https://huggingface.co/${model.repo}/resolve/${encodeURIComponent(model.revision)}/${encodeURIComponent(model.filename)}?download=true`;
}

async function isCompleteModelFile(file: string, model: ManagedModel): Promise<boolean> {
	try {
		const info = await stat(file);
		return info.isFile() && info.size === model.size;
	} catch {
		return false;
	}
}

async function ensureModel(model: ManagedModel, onStatus?: StatusCallback): Promise<string> {
	const file = modelCachePath(model);
	if (await isCompleteModelFile(file, model)) return file;

	try {
		const info = await stat(file);
		if (info.isFile() && info.size !== model.size) {
			const bad = `${file}.bad-${Date.now()}`;
			await rename(file, bad);
			await appendLog(`[${new Date().toISOString()}] moved wrong-size model ${file} (${info.size} bytes) to ${bad}\n`);
		}
	} catch {}

	const partial = `${file}.part`;
	try {
		const partialInfo = await stat(partial);
		if (partialInfo.isFile() && partialInfo.size === model.size) await rename(partial, file);
	} catch {}
	if (await isCompleteModelFile(file, model)) return file;

	onStatus?.(`downloading ${model.quant} model (${formatBytes(model.size)})`);
	await downloadFile(huggingFaceModelUrl(model), file, `downloading ${model.filename}`, onStatus);

	if (!(await isCompleteModelFile(file, model))) {
		const info = await stat(file).catch(() => undefined);
		throw new Error(
			`Downloaded ${model.filename} but size was ${info ? formatBytes(info.size) : "missing"}; expected ${formatBytes(model.size)}`,
		);
	}
	await writeFile(`${file}.sha256`, `${model.sha256}  ${model.filename}\n`, "utf8").catch(() => {});
	return file;
}

async function isLockStale(): Promise<boolean> {
	const owner = await readJson<{ pid?: number; processStart?: string }>(join(LOCK_DIR, "owner.json"));
	if (owner?.pid) {
		if (!isPidAlive(owner.pid)) return true;
		if (owner.processStart) {
			const currentStart = await processStart(owner.pid);
			if (currentStart && currentStart !== owner.processStart) return true;
		}
	}

	try {
		const info = await stat(LOCK_DIR);
		return Date.now() - info.mtimeMs > LOCK_STALE_MS;
	} catch {
		return true;
	}
}

async function withLock<T>(fn: () => Promise<T>, timeoutMs = LOCK_TIMEOUT_MS, abortOnDispose = false): Promise<T> {
	await mkdir(LLAMACPP_DIR, { recursive: true });
	const started = Date.now();

	while (true) {
		if (abortOnDispose && (runtimeDisposed || shuttingDown)) throw new Error("llama.cpp startup cancelled");
		try {
			await mkdir(LOCK_DIR);
			await writeJsonAtomic(join(LOCK_DIR, "owner.json"), {
				managedBy: MANAGED_BY,
				pid: process.pid,
				processStart: await getOwnProcessStart(),
				createdAt: Date.now(),
			});
			break;
		} catch (error: any) {
			if (error?.code !== "EEXIST") throw error;
			if (await isLockStale()) {
				await rm(LOCK_DIR, { recursive: true, force: true });
				continue;
			}
			if (timeoutMs > 0 && Date.now() - started > timeoutMs) {
				throw new Error(`Timed out waiting for llama.cpp lifecycle lock at ${LOCK_DIR}`);
			}
			await sleep(100 + Math.floor(Math.random() * 150));
		}
	}

	try {
		return await fn();
	} finally {
		await rm(LOCK_DIR, { recursive: true, force: true });
	}
}

async function touchLease(model = activeLeaseModel): Promise<void> {
	if (!model) return;
	const now = Date.now();
	const lease: Lease = {
		managedBy: MANAGED_BY,
		usesLlamaCpp: true,
		pid: process.pid,
		processStart: await getOwnProcessStart(),
		cwd: process.cwd(),
		modelId: model.id,
		quant: model.quant,
		startedAt: leaseStartedAt,
		updatedAt: now,
		updatedAtIso: new Date(now).toISOString(),
	};
	await writeJsonAtomic(LEASE_FILE, lease);
}

function startHeartbeat(): void {
	if (heartbeat) clearInterval(heartbeat);
	heartbeat = setInterval(() => {
		void touchLease().catch(() => {});
	}, HEARTBEAT_MS);
	heartbeat.unref?.();
}

function stopHeartbeat(): void {
	if (heartbeat) {
		clearInterval(heartbeat);
		heartbeat = undefined;
	}
}

async function removeOwnLease(): Promise<void> {
	await removeFile(LEASE_FILE);
	leaseActive = false;
}

async function pruneLeases(): Promise<void> {
	await mkdir(CLIENT_DIR, { recursive: true });
	const entries = await readdir(CLIENT_DIR).catch(() => [] as string[]);
	const now = Date.now();

	for (const entry of entries) {
		if (!entry.endsWith(".json")) continue;
		const file = join(CLIENT_DIR, entry);
		const [lease, info] = await Promise.all([readJson<Lease>(file), stat(file).catch(() => undefined)]);
		const staleByAge = !info || now - info.mtimeMs > LEASE_TTL_MS;
		const staleByProcess = !(await isLeaseForLiveProcess(lease));
		if (staleByAge || staleByProcess) await removeFile(file);
	}
}

async function readLiveLeases(): Promise<Lease[]> {
	await pruneLeases();
	const entries = await readdir(CLIENT_DIR).catch(() => [] as string[]);
	const leases: Lease[] = [];
	for (const entry of entries) {
		if (!entry.endsWith(".json")) continue;
		const lease = await readJson<Lease>(join(CLIENT_DIR, entry));
		if (await isLeaseForLiveProcess(lease)) leases.push(lease!);
	}
	return leases;
}

async function activateLease(model: ManagedModel): Promise<void> {
	await ensureDirs();
	activeLeaseModel = model;
	await touchLease(model);
	leaseActive = true;
	await pruneLeases();
	await ensureWatchdog();
	startHeartbeat();
}

async function readState(): Promise<ServerState | undefined> {
	return readJson<ServerState>(STATE_FILE);
}

async function clearState(): Promise<void> {
	await removeFile(STATE_FILE);
}

async function fetchServerModelIds(baseUrl: string): Promise<string[] | undefined> {
	const controller = new AbortController();
	const timeout = setTimeout(() => controller.abort(), HTTP_CHECK_TIMEOUT_MS);
	try {
		const response = await fetch(`${baseUrl}/models`, { signal: controller.signal });
		if (!response.ok) return undefined;
		const payload = (await response.json()) as { data?: Array<{ id?: string }> };
		return payload.data?.map((item) => item.id).filter((id): id is string => typeof id === "string");
	} catch {
		return undefined;
	} finally {
		clearTimeout(timeout);
	}
}

async function checkHttpReady(modelId?: string, baseUrl?: string): Promise<boolean> {
	baseUrl ??= (await readState())?.baseUrl;
	if (!baseUrl) return false;
	const modelIds = await fetchServerModelIds(baseUrl);
	if (!modelIds) return false;
	return modelId ? modelIds.includes(modelId) : true;
}

async function waitForPidExit(pid: number, timeoutMs: number): Promise<boolean> {
	const deadline = Date.now() + timeoutMs;
	while (Date.now() < deadline) {
		if (!isPidAlive(pid)) return true;
		await sleep(500);
	}
	return !isPidAlive(pid);
}

async function waitForServerReady(model: ManagedModel, onStatus?: StatusCallback): Promise<void> {
	const started = Date.now();
	let lastStatus = 0;

	while (Date.now() - started < READY_TIMEOUT_MS) {
		if (runtimeDisposed || shuttingDown) return;
		const state = await readState();
		if (state?.baseUrl && (await checkHttpReady(model.id, state.baseUrl))) return;

		if (state?.pid && !isPidAlive(state.pid)) {
			throw new Error(`llama-server exited before becoming ready; see ${LOG_FILE}`);
		}

		if (Date.now() - lastStatus > 10_000) {
			const elapsed = Math.round((Date.now() - started) / 1000);
			onStatus?.(`llama-server starting ${model.quant} (${elapsed}s)`);
			lastStatus = Date.now();
		}
		await sleep(1_000);
	}

	const state = await readState();
	throw new Error(`Timed out waiting for llama-server at ${state?.baseUrl ?? describeApiBaseUrl()}; see ${LOG_FILE}`);
}

async function writeAdoptedServerStateLocked(pid: number, model: ManagedModel, baseUrl: string, port: number): Promise<void> {
	const args = await processArgs(pid);
	const now = Date.now();
	const binary = args?.split(/\s+/, 1)[0] || "llama-server";
	const state: ServerState = {
		managedBy: MANAGED_BY,
		pid,
		baseUrl,
		port,
		cwd: process.cwd(),
		binary,
		args: args ? [args] : [],
		modelId: model.id,
		modelName: model.name,
		quant: model.quant,
		modelFile: modelCachePath(model),
		releaseTag: LLAMACPP_RELEASE_TAG,
		assetName: resolvedRuntimeAsset?.name,
		startedAt: now,
		startedAtIso: new Date(now).toISOString(),
	};
	await writeJsonAtomic(STATE_FILE, state);
	await appendLog(`\n[${new Date().toISOString()}] adopted existing llama-server pid=${pid} model=${model.id}\n`);
}

function serverArgs(model: ManagedModel, modelFile: string, port: number): string[] {
	const args = [
		"--host",
		LLAMACPP_HOST,
		"--port",
		String(port),
		"-m",
		modelFile,
		"-a",
		model.id,
		"--ctx-size",
		String(DEFAULT_CTX_SIZE),
		"--jinja",
		"--reasoning-format",
		"deepseek",
		"--reasoning",
		"auto",
	];

	const mtpSetting = process.env.LLAMACPP_ENABLE_MTP?.toLowerCase();
	const enableMtp = mtpSetting !== "0" && mtpSetting !== "false" && mtpSetting !== "no";

	const parallel = process.env.LLAMACPP_PARALLEL;
	if (parallel) args.push("--parallel", parallel);
	else if (enableMtp) args.push("--parallel", "1");

	const gpuLayers = process.env.LLAMACPP_GPU_LAYERS;
	if (gpuLayers) args.push("--n-gpu-layers", gpuLayers);

	if (enableMtp) args.push("--spec-type", "mtp", "--spec-draft-n-max", process.env.LLAMACPP_MTP_DRAFT_N_MAX ?? "3");

	const extra = process.env.LLAMACPP_SERVER_ARGS;
	if (extra) args.push(...extra.split(/\s+/).filter(Boolean));

	return args;
}

async function startServerLocked(runtimeDir: string, model: ManagedModel, modelFile: string): Promise<void> {
	const binary = process.env.LLAMACPP_SERVER_BINARY ? resolve(process.env.LLAMACPP_SERVER_BINARY) : await findLlamaServerBinary(runtimeDir);
	if (!binary) throw new Error(`Cannot find llama-server in ${runtimeDir}`);
	try {
		await access(binary, constants.X_OK);
	} catch {
		throw new Error(`Cannot execute llama-server at ${binary}`);
	}

	const port = await chooseServerPort();
	const existingPid = await findListeningPid(port);
	if (existingPid) throw new Error(`Port ${port} is already in use by ${await describeProcess(existingPid)}; cannot start llama-server`);
	const baseUrl = apiBaseUrlForPort(port);
	const args = serverArgs(model, modelFile, port);
	await appendLog(`\n[${new Date().toISOString()}] start llama-server (${model.id}) on ${baseUrl}\n$ ${[binary, ...args].map(shellQuote).join(" ")}\n`);
	const logFd = openSync(LOG_FILE, "a");
	let childPid: number | undefined;
	try {
		const child = spawn(binary, args, {
			cwd: runtimeDir,
			detached: true,
			stdio: ["ignore", logFd, logFd],
			env: process.env,
		});
		child.unref();
		childPid = child.pid;
	} finally {
		closeSync(logFd);
	}

	if (!childPid) throw new Error("Failed to start llama-server: no child PID");

	const now = Date.now();
	const state: ServerState = {
		managedBy: MANAGED_BY,
		pid: childPid,
		baseUrl,
		port,
		cwd: runtimeDir,
		binary,
		args,
		modelId: model.id,
		modelName: model.name,
		quant: model.quant,
		modelFile,
		releaseTag: LLAMACPP_RELEASE_TAG,
		assetName: resolvedRuntimeAsset?.name,
		startedAt: now,
		startedAtIso: new Date(now).toISOString(),
	};
	await writeJsonAtomic(STATE_FILE, state);
}

async function stopServerLocked(pid: number, reason: string): Promise<void> {
	await appendLog(`\n[${new Date().toISOString()}] stop llama-server pid=${pid}: ${reason}\n`);
	const state = await readState();
	if (state?.pid === pid) {
		await writeJsonAtomic(STATE_FILE, {
			...state,
			stopping: true,
			stoppingAt: Date.now(),
			stoppingAtIso: new Date().toISOString(),
		});
	}
	try {
		process.kill(pid, "SIGTERM");
	} catch {}
	if (!(await waitForPidExit(pid, SHUTDOWN_GRACE_MS))) {
		try {
			process.kill(pid, "SIGKILL");
		} catch {}
		await waitForPidExit(pid, 2_000);
	}
	if (!isPidAlive(pid)) await clearState();
}

function describeLease(lease: Lease): string {
	return `pid ${lease.pid} (${lease.modelId}, updated ${lease.updatedAtIso})`;
}

async function ensureServerManagedInner(model: ManagedModel, onStatus?: StatusCallback): Promise<void> {
	if (runtimeDisposed || shuttingDown) return;
	let stoppingPid: number | undefined;

	await withLock(async () => {
		await activateLease(model);
		if (runtimeDisposed || shuttingDown) return;
		await touchLease(model);
		await pruneLeases();

		const state = await readState();
		if (state?.managedBy === MANAGED_BY && state.pid && isPidAlive(state.pid) && (await looksLikeLlamaServer(state.pid))) {
			if (state.modelId === model.id) {
				if (state.stopping) stoppingPid = state.pid;
				return;
			}

			const blockers = (await readLiveLeases()).filter((lease) => lease.pid !== process.pid && lease.modelId !== model.id);
			if (blockers.length > 0) {
				throw new Error(
					`llama-server is running ${state.modelId}, but ${model.id} was requested. Active other leases: ${blockers
						.map(describeLease)
						.join(", ")}`,
				);
			}
			await stopServerLocked(state.pid, `switch from ${state.modelId} to ${model.id}`);
		} else if (state?.pid) {
			await clearState();
		}

		if (LLAMACPP_PORT) {
			const fixedBaseUrl = apiBaseUrlForPort(LLAMACPP_PORT);
			if (await checkHttpReady(model.id, fixedBaseUrl)) {
				const pid = await findListeningLlamaServerPid(LLAMACPP_PORT);
				if (pid) await writeAdoptedServerStateLocked(pid, model, fixedBaseUrl, LLAMACPP_PORT);
				return;
			}

			const listeningPid = await findListeningPid(LLAMACPP_PORT);
			if (listeningPid) {
				if (await looksLikeLlamaServer(listeningPid)) {
					const ids = await fetchServerModelIds(fixedBaseUrl);
					throw new Error(
						`Port ${LLAMACPP_PORT} is already used by llama-server pid ${listeningPid} with models ${ids?.join(", ") || "unknown"}; cannot start ${model.id}`,
					);
				}
				throw new Error(`Port ${LLAMACPP_PORT} is already in use by ${await describeProcess(listeningPid)}; cannot start llama-server`);
			}
		}
		if (runtimeDisposed || shuttingDown) return;

		const runtimeDir = await ensureRuntime(onStatus);
		if (runtimeDisposed || shuttingDown) return;
		const modelFile = await ensureModel(model, onStatus);
		if (runtimeDisposed || shuttingDown) return;

		onStatus?.(`starting llama-server (${model.quant})`);
		await startServerLocked(runtimeDir, model, modelFile);
	}, STARTUP_LOCK_TIMEOUT_MS, true);

	if (runtimeDisposed || shuttingDown) return;

	if (stoppingPid) {
		onStatus?.("waiting for previous llama-server shutdown");
		if (!(await waitForPidExit(stoppingPid, SHUTDOWN_GRACE_MS))) {
			throw new Error(`Previous llama-server pid ${stoppingPid} did not exit`);
		}
		await withLock(async () => {
			const state = await readState();
			if (state?.pid === stoppingPid && !isPidAlive(stoppingPid)) await clearState();
		}, LOCK_TIMEOUT_MS);
		return ensureServerManagedInner(model, onStatus);
	}

	await waitForServerReady(model, onStatus);
}

function ensureServerManaged(model: ManagedModel, onStatus?: StatusCallback): Promise<void> {
	if (!startupPromise) {
		startupPromise = ensureServerManagedInner(model, onStatus).finally(() => {
			startupPromise = undefined;
		});
	}
	return startupPromise;
}

function createStartupStatusCallback(ctx: ExtensionContext | undefined, notify: boolean): StatusCallback {
	let lastNotification: string | undefined;
	return (message) => {
		if (!message) return;
		void appendLog(`[${new Date().toISOString()}] ${message}\n`).catch(() => {});

		if (!notify || !ctx?.hasUI || message === lastNotification) return;
		if (/^llama-server starting .*\(\d+s\)$/.test(message)) return;
		lastNotification = message;
		ctx.ui.notify(message, "info");
	};
}

async function stopServerIfUnused(): Promise<void> {
	// The watchdog owns lease refcounting and server shutdown. Keep /quit fast:
	// removing our lease is enough for it to stop llama-server when nobody else uses it.
	await removeOwnLease();
}

async function statusSummary(): Promise<string> {
	const state = await readState();
	const lines = [`llama.cpp cache: ${LLAMACPP_DIR}`, `provider: ${PROVIDER_ID} (${describeApiBaseUrl()})`];
	if (state?.pid && isPidAlive(state.pid)) {
		lines.push(`server: pid ${state.pid}, ${state.baseUrl}, model ${state.modelId}, ${state.quant}, ${state.stopping ? "stopping" : "running"}`);
	} else {
		lines.push("server: not running");
	}
	lines.push(`models: ${MODELS.map((model) => model.id).join(", ")}`);
	lines.push(`log: ${LOG_FILE}`);
	return lines.join("\n");
}

function registerLlamaCppCommand(pi: ExtensionAPI): void {
	pi.registerCommand("llamacpp", {
		description: "Show llama.cpp status/log",
		handler: async (args, ctx) => {
			const action = args.trim().toLowerCase();
			if (action === "status") {
				ctx.ui.notify(await statusSummary(), "info");
				return;
			}

			if (action === "stop") {
				stopHeartbeat();
				activeLeaseModel = undefined;
				await withLock(async () => {
					await removeOwnLease();
					const leases = (await readLiveLeases()).filter((lease) => lease.pid !== process.pid);
					const state = await readState();
					if (leases.length > 0) throw new Error(`Cannot stop llama-server; active leases: ${leases.map(describeLease).join(", ")}`);
					if (state?.pid && isPidAlive(state.pid) && (await looksLikeLlamaServer(state.pid))) {
						await stopServerLocked(state.pid, "/llamacpp stop");
					}
				}, LOCK_TIMEOUT_MS);
				ctx.ui.notify("llama-server stopped", "info");
				return;
			}

			if (!ctx.hasUI) {
				ctx.ui.notify(`llama.cpp log: ${LOG_FILE}`, "info");
				return;
			}

			let viewer: LlamaCppLogViewer | undefined;
			try {
				await ctx.ui.custom<void>(
					(tui, theme, _keybindings, done) => {
						viewer = new LlamaCppLogViewer(tui, theme, done);
						return viewer;
					},
					{
						overlay: true,
						overlayOptions: {
							width: "90%",
							minWidth: 60,
							maxHeight: "85%",
							anchor: "center",
							margin: 1,
						},
					},
				);
			} finally {
				viewer?.dispose();
			}
		},
	});
}

class LocalAssistantMessageEventStream {
	private queue: any[] = [];
	private waiting: Array<(result: IteratorResult<any>) => void> = [];
	private done = false;
	private resolveFinalResult!: (result: any) => void;
	private finalResultPromise = new Promise<any>((resolve) => {
		this.resolveFinalResult = resolve;
	});

	push(event: any): void {
		if (this.done) return;
		if (event.type === "done" || event.type === "error") {
			this.done = true;
			this.resolveFinalResult(event.type === "done" ? event.message : event.error);
		}
		const waiter = this.waiting.shift();
		if (waiter) waiter({ value: event, done: false });
		else this.queue.push(event);
	}

	end(result?: any): void {
		this.done = true;
		if (result !== undefined) this.resolveFinalResult(result);
		while (this.waiting.length > 0) this.waiting.shift()?.({ value: undefined, done: true });
	}

	async *[Symbol.asyncIterator](): AsyncGenerator<any> {
		while (true) {
			if (this.queue.length > 0) yield this.queue.shift();
			else if (this.done) return;
			else {
				const result = await new Promise<IteratorResult<any>>((resolve) => this.waiting.push(resolve));
				if (result.done) return;
				yield result.value;
			}
		}
	}

	result(): Promise<any> {
		return this.finalResultPromise;
	}
}

function createLocalAssistantMessageEventStream(): LocalAssistantMessageEventStream {
	return new LocalAssistantMessageEventStream();
}

function errorAssistantMessage(model: Model<any>, error: unknown, aborted = false): AssistantMessage {
	return {
		role: "assistant",
		content: [],
		api: model.api,
		provider: model.provider,
		model: model.id,
		usage: {
			input: 0,
			output: 0,
			cacheRead: 0,
			cacheWrite: 0,
			totalTokens: 0,
			cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
		},
		stopReason: aborted ? "aborted" : "error",
		errorMessage: describeError(error),
		timestamp: Date.now(),
	};
}

function applyLlamaCppPayloadDefaults(payload: unknown, reasoningEnabled: boolean): Record<string, any> {
	const base = payload && typeof payload === "object" && !Array.isArray(payload) ? (payload as Record<string, any>) : {};
	const chatTemplateKwargs =
		base.chat_template_kwargs && typeof base.chat_template_kwargs === "object" && !Array.isArray(base.chat_template_kwargs)
			? (base.chat_template_kwargs as Record<string, any>)
			: {};

	return {
		...(reasoningEnabled ? QWEN_THINKING_SAMPLING : QWEN_INSTRUCT_SAMPLING),
		...base,
		chat_template_kwargs: {
			enable_thinking: reasoningEnabled,
			preserve_thinking: true,
			...chatTemplateKwargs,
		},
	};
}

function streamLlamaCpp(model: Model<any>, context: Context, options?: SimpleStreamOptions) {
	const stream = createLocalAssistantMessageEventStream();
	void (async () => {
		try {
			const managedModel = MODEL_BY_ID.get(model.id);
			if (!managedModel) throw new Error(`Unknown llama.cpp model: ${model.id}`);

			const alreadyReady = await checkHttpReady(managedModel.id);
			const status = createStartupStatusCallback(activeProviderContext, !alreadyReady);
			if (!alreadyReady) status(`preparing llama.cpp (${managedModel.quant})`);
			await ensureServerManaged(managedModel, status);
			if (!alreadyReady) status(`llama-server ready (${managedModel.quant})`);
			const state = await readState();
			if (!state?.baseUrl || state.modelId !== managedModel.id || !isPidAlive(state.pid)) {
				throw new Error(`llama-server state is not ready for ${managedModel.id}`);
			}

			const runtimeModel = { ...model, baseUrl: state.baseUrl } as Model<"openai-completions">;
			const inner = streamOpenAICompletions(runtimeModel, context, {
				...(options as any),
				apiKey: options?.apiKey ?? API_KEY,
				reasoningEffort: options?.reasoning,
				onPayload: async (payload: unknown) => {
					const llamaPayload = applyLlamaCppPayloadDefaults(payload, !!options?.reasoning);
					const nextPayload = await options?.onPayload?.(llamaPayload, model);
					return nextPayload === undefined ? llamaPayload : nextPayload;
				},
				onResponse: (response: any) => options?.onResponse?.(response, model),
			} as any);

			for await (const event of inner as any) stream.push(event);
			stream.end();
		} catch (error) {
			const message = errorAssistantMessage(model, error, !!options?.signal?.aborted);
			stream.push({ type: "error", reason: message.stopReason as "error" | "aborted", error: message });
			stream.end();
		}
	})();
	return stream;
}

function registerLlamaCppProvider(pi: ExtensionAPI): void {
	pi.registerProvider(PROVIDER_ID, {
		name: "llama.cpp local",
		baseUrl: PROVIDER_BASE_URL,
		api: "openai-completions",
		apiKey: API_KEY,
		streamSimple: streamLlamaCpp,
		compat: {
			supportsStore: false,
			supportsDeveloperRole: false,
			supportsReasoningEffort: false,
			supportsUsageInStreaming: true,
			maxTokensField: "max_tokens",
			supportsStrictMode: false,
			thinkingFormat: "qwen-chat-template",
		},
		models: MODELS.map((model) => ({
			id: model.id,
			name: model.name,
			reasoning: true,
			thinkingLevelMap: {
				minimal: null,
				low: null,
				medium: null,
				high: "enabled",
				xhigh: null,
			},
			input: ["text"],
			contextWindow: DEFAULT_CTX_SIZE,
			maxTokens: DEFAULT_MAX_TOKENS,
			cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
		})),
	} as any);
}

export default function (pi: ExtensionAPI) {
	runtimeDisposed = false;
	shuttingDown = false;
	leaseStartedAt = Date.now();
	leaseActive = false;
	activeLeaseModel = undefined;
	watchdogStarted = false;
	startupPromise = undefined;
	activeSetupChild = undefined;
	activeProviderContext = undefined;
	resolvedRuntimeDir = undefined;
	resolvedRuntimeAsset = undefined;

	registerLlamaCppProvider(pi);
	registerLlamaCppCommand(pi);

	pi.on("session_start", async (_event, ctx) => {
		activeProviderContext = ctx;
	});

	pi.on("agent_start", async (_event, ctx) => {
		if (ctx.model?.provider === PROVIDER_ID) activeProviderContext = ctx;
	});

	pi.on("turn_start", async (_event, ctx) => {
		if (ctx.model?.provider === PROVIDER_ID) activeProviderContext = ctx;
	});

	pi.on("before_provider_request", async (_event, ctx) => {
		if (ctx.model?.provider !== PROVIDER_ID) return;
		activeProviderContext = ctx;
		const model = MODEL_BY_ID.get(ctx.model.id);
		if (!model) return;

		const alreadyReady = await checkHttpReady(model.id);
		let lastNotification: string | undefined;
		const notifyStatus: StatusCallback | undefined = alreadyReady
			? undefined
			: (message) => {
					if (!message || message === lastNotification) return;
					if (/^llama-server starting .*\(\d+s\)$/.test(message)) return;
					lastNotification = message;
					ctx.ui.notify(message, "info");
				};

		try {
			notifyStatus?.(`preparing llama.cpp (${model.quant})`);
			await ensureServerManaged(model, notifyStatus);
			if (!alreadyReady) ctx.ui.notify(`llama-server ready (${model.quant})`, "info");
		} catch (error) {
			ctx.ui.notify(`llama-server startup failed: ${describeError(error)}`, "error");
			throw error;
		}
	});

	pi.on("session_shutdown", async (event, ctx) => {
		activeProviderContext = undefined;
		runtimeDisposed = true;
		stopHeartbeat();
		killActiveSetupChild();

		try {
			if (startupPromise) await Promise.race([startupPromise.catch(() => {}), sleep(5_000)]);
		} catch {}

		// Session switches and /reload immediately create another extension instance
		// in the same pi process. Keep the lease for those hand-offs.
		if (event.reason !== "quit") return;

		shuttingDown = true;
		try {
			await stopServerIfUnused();
		} catch (error) {
			if (!isLockTimeout(error)) ctx.ui.notify(`llama-server shutdown failed: ${describeError(error)}`, "error");
		}
	});
}
