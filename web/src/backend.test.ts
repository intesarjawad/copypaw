import { describe, expect, it } from "vitest";
import { BackendError, LocalBackend, TOKEN_HEADER, errorDetail } from "./backend";
import { JOINT_IDS } from "./joints";

interface Sent {
  readonly url: string;
  readonly init: RequestInit | undefined;
}

function fakeFetch(status: number, body: string, sent: Sent[]): typeof fetch {
  return (input, init) => {
    sent.push({ url: input instanceof Request ? input.url : String(input), init });
    return Promise.resolve(new Response(body, { status }));
  };
}

function header(sent: Sent | undefined, name: string): string | null {
  return new Headers(sent?.init?.headers).get(name);
}

const RIG = {
  rig: "r1",
  width: 300,
  height: 200,
  box_height: 140,
  outline: [
    [40, 30],
    [260, 30],
    [260, 170],
  ],
  joints: Object.fromEntries(JOINT_IDS.map((id, index) => [id, [index, index + 1]])),
};

describe("LocalBackend", () => {
  it("sends the token and parses the rig", async () => {
    const sent: Sent[] = [];
    const backend = new LocalBackend("secret", fakeFetch(200, JSON.stringify(RIG), sent));

    const rig = await backend.rig(new Blob(["png"]));

    expect(sent[0]?.url).toBe("/api/rig");
    expect(sent[0]?.init?.method).toBe("POST");
    expect(header(sent[0], TOKEN_HEADER)).toBe("secret");
    expect(rig.id).toBe("r1");
    expect(rig.boxHeight).toBe(140);
    expect(rig.outline[1]).toEqual({ x: 260, y: 30 });
    expect(rig.joints.neck).toEqual({ x: 3, y: 4 });
  });

  it("sends a gallery choice and the credit flag", async () => {
    const sent: Sent[] = [];
    const backend = new LocalBackend("t", fakeFetch(200, '{"job": "j1"}', sent));

    const job = await backend.animate("r1", { kind: "gallery", id: "walk" }, false);

    expect(job).toBe("j1");
    const body = sent[0]?.init?.body;
    expect(typeof body === "string" ? JSON.parse(body) : body).toEqual({
      rig: "r1",
      credit: false,
      gallery: "walk",
    });
  });

  it("raises the server's message for a failed request", async () => {
    const error = { code: "joints_invalid", what: "Bad joints.", next: "Fix them." };
    const backend = new LocalBackend("t", fakeFetch(400, JSON.stringify(error), []));

    const failure: unknown = await backend.job("j1").catch((caught: unknown) => caught);

    expect(failure).toBeInstanceOf(BackendError);
    expect(errorDetail(failure)).toEqual(error);
  });

  it("keeps the server's next step for a missing token", async () => {
    const error = {
      code: "app_request",
      what: "The page sent a request Copypaw cannot use: the session token is missing or wrong.",
      next: "Reload the address `copypaw app` printed and start again.",
    };
    const backend = new LocalBackend("t", fakeFetch(403, JSON.stringify(error), []));

    const failure: unknown = await backend.health().catch((caught: unknown) => caught);

    expect(errorDetail(failure)).toEqual(error);
  });

  it("falls back to the internal message when the reply is not JSON", async () => {
    const backend = new LocalBackend("t", fakeFetch(502, "Bad Gateway", []));

    const failure: unknown = await backend.health().catch((caught: unknown) => caught);

    expect(errorDetail(failure).what).toBe("Copypaw hit a bug it did not expect.");
  });

  it("parses a finished job", async () => {
    const results = {
      video: "/v",
      gif: "/g",
      mp4: "/m",
      bvh: "/b",
      character: "/c",
      clip: "/k",
      skeleton: "/s",
    };
    const body = JSON.stringify({ state: "done", stage: "done", results, timing: null });
    const backend = new LocalBackend("t", fakeFetch(200, body, []));

    expect(await backend.job("j1")).toEqual({
      state: "done",
      results: { kind: "files", ...results },
      timing: null,
    });
  });
});
