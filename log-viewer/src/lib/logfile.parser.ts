import { z } from 'zod';
import type { DebugLog, Span, TaskSpan } from './log';

const logEntry = z.object({
	parent: z.string(),
	message: z.string(),
	value: z.any(),
	timestamp: z.string()
});

type LogEntry = z.infer<typeof logEntry>;

const spanStart = z.object({
	uuid: z.string(),
	parent: z.string(),
	name: z.string(),
	start: z.string()
});

type SpanStart = z.infer<typeof spanStart>;

const spanEnd = z.object({
	uuid: z.string(),
	end: z.string()
});

type SpanEnd = z.infer<typeof spanEnd>;

const taskStart = z.object({
	uuid: z.string(),
	parent: z.string(),
	name: z.string(),
	start: z.string(),
	input: z.any()
});

type TaskStart = z.infer<typeof taskStart>;

const taskEnd = z.object({
	uuid: z.string(),
	end: z.string(),
	output: z.any()
});

type TaskEnd = z.infer<typeof taskEnd>;

const logLine = z.discriminatedUnion('entry_type', [
	z.object({
		entry_type: z.literal('LogEntry'),
		entry: logEntry
	}),
	z.object({
		entry_type: z.literal('StartSpan'),
		entry: spanStart
	}),
	z.object({
		entry_type: z.literal('EndSpan'),
		entry: spanEnd
	}),
	z.object({
		entry_type: z.literal('StartTask'),
		entry: taskStart
	}),
	z.object({
		entry_type: z.literal('EndTask'),
		entry: taskEnd
	})
]);

export type LogLine = z.infer<typeof logLine>;

export function parseLogLines(lines: LogLine[]): DebugLog {
	const builder = new LogBuilder();
	for (const line of lines) {
		switch (line.entry_type) {
			case 'LogEntry':
				builder.addLogEntry(line.entry);
				break;
			case 'StartSpan':
				builder.startSpan(line.entry);
				break;
			case 'EndSpan':
				builder.endSpan(line.entry);
				break;
			case 'StartTask':
				builder.startTask(line.entry);
				break;
			case 'EndTask':
				builder.endTask(line.entry);
				break;
		}
	}
	return builder.root;
}

class LogBuilder {
	root: DebugLog = { name: '', logs: [] };
	private loggers: Map<string, DebugLog> = new Map<string, DebugLog>();
	private spans: Map<string, Span> = new Map<string, Span>();
	private tasks: Map<string, TaskSpan> = new Map<string, TaskSpan>();

	addLogEntry(entry: LogEntry) {
		const parent = this.parentLogger(entry.parent);
		// entry.value is any, but value is "Json"
		// eslint-disable-next-line @typescript-eslint/no-unsafe-assignment
		parent.logs.push({ message: entry.message, value: entry.value, timestamp: entry.timestamp });
	}

	startSpan(entry: SpanStart) {
		const parent = this.parentLogger(entry.parent);
		const span: Span = {
			name: entry.name,
			start_timestamp: entry.start,
			end_timestamp: entry.start,
			logs: []
		};
		parent.logs.push(span);
		this.spans.set(entry.uuid, span);
		this.loggers.set(entry.uuid, span);
	}

	endSpan(entry: SpanEnd) {
		const span = this.spans.get(entry.uuid);
		span!.end_timestamp = entry.end;
	}

	startTask(entry: TaskStart) {
		const parent = this.parentLogger(entry.parent);
		const task: TaskSpan = {
			name: entry.name,
			start_timestamp: entry.start,
			end_timestamp: entry.start,
			// eslint-disable-next-line @typescript-eslint/no-unsafe-assignment
			input: entry.input,
			output: undefined,
			logs: []
		};
		parent.logs.push(task);
		this.tasks.set(entry.uuid, task);
		this.loggers.set(entry.uuid, task);
	}

	endTask(entry: TaskEnd) {
		const task = this.tasks.get(entry.uuid);
		task!.end_timestamp = entry.end;
		// eslint-disable-next-line @typescript-eslint/no-unsafe-assignment
		task!.output = entry.output;
	}

	parentLogger(uuid: string): DebugLog {
		const parent = this.loggers.get(uuid);
		if (parent) {
			return parent;
		}
		this.root = { name: uuid, logs: [] };
		this.loggers.set(uuid, this.root);
		return this.root;
	}
}
