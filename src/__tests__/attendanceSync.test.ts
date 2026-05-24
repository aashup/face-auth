const mockPost = jest.fn();
jest.mock(
  'axios',
  () => ({
    __esModule: true,
    default: {
      create: () => ({
        post: mockPost,
        interceptors: { request: { use: jest.fn() }, response: { use: jest.fn() } },
      }),
    },
  }),
  { virtual: true },
);

import { syncAttendance } from '../sync/attendanceSync';
import { append, openAuditLog, pendingCount, type AttendanceEvent } from '../secure/auditLog';
import { __resetAsyncStorage } from '../../__mocks__/async-storage';
import { __resetKeychain } from '../../__mocks__/react-native-keychain';

function ev(id: string): AttendanceEvent {
  return {
    id,
    ts: Number(id),
    ok: true,
    matchScore: 0.9,
    livenessScore: 0.95,
    challengePassed: true,
    deviceId: 'dev-1',
  };
}

const OPTS = { awsSyncUrl: 'https://x', deviceToken: 't', purgeAfterSync: true };

describe('attendance sync + purge', () => {
  beforeEach(async () => {
    __resetAsyncStorage();
    __resetKeychain();
    mockPost.mockReset();
    await openAuditLog();
    append(ev('1'));
    append(ev('2'));
    append(ev('3'));
  });

  it('uploads and purges all acked events', async () => {
    mockPost.mockImplementation(async (_url, body) => ({
      data: { acked: body.events.map((e: AttendanceEvent) => e.id) },
    }));
    const out = await syncAttendance(OPTS);
    expect(out.ok).toBe(true);
    expect(out.uploaded).toBe(3);
    expect(out.remaining).toBe(0);
    expect(pendingCount()).toBe(0);
  });

  it('keeps unacked events on partial ack', async () => {
    mockPost.mockImplementation(async (_url, body) => ({
      data: { acked: [body.events[0].id] },
    }));
    const out = await syncAttendance(OPTS);
    expect(out.uploaded).toBe(1);
    expect(pendingCount()).toBe(2);
  });

  it('purges nothing when the server acks none', async () => {
    mockPost.mockResolvedValue({ data: { acked: [] } });
    const out = await syncAttendance(OPTS);
    expect(out.ok).toBe(false);
    expect(pendingCount()).toBe(3);
  });

  it('reports failure and keeps events on network error', async () => {
    mockPost.mockRejectedValue(new Error('offline'));
    const out = await syncAttendance(OPTS);
    expect(out.ok).toBe(false);
    expect(out.error).toBe('offline');
    expect(pendingCount()).toBe(3);
  });
});
