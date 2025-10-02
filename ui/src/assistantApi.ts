export async function assistantGenerate(req: {
	kind: "flowchart" | "short_notes" | "detailed_notes" | "timeline" | "key_insights" | "flashcards";
	text?: string;
	file?: { id: string; name?: string };
	format?: "md" | "txt" | "docx" | "pdf";
	auth_token?: string;
}) {
	// Go through gateway to reuse CORS/config and consistent host
	const res = await fetch("http://127.0.0.1:4000/api/assistant_generate", {
		method: "POST",
		headers: { "Content-Type": "application/json" },
		body: JSON.stringify(req),
	});
	if (!res.ok) {
		throw new Error(await res.text());
	}
	return (await res.json()) as { kind: string; filename: string; mime: string; base64: string; content: any };
}

export function downloadFromResponse(r: { filename: string; mime: string; base64: string }) {
	const b = atob(r.base64);
	const bytes = new Uint8Array(b.length);
	for (let i = 0; i < b.length; i++) bytes[i] = b.charCodeAt(i);
	const blob = new Blob([bytes], { type: r.mime || "application/octet-stream" });
	const a = document.createElement("a");
	a.href = URL.createObjectURL(blob);
	a.download = r.filename || "output.md";
	a.click();
	URL.revokeObjectURL(a.href);
}


