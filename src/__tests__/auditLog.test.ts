import {
  append,
  openAuditLog,
  pending,
  pendingCount,
  purge,
  verifyChain,
  type AttendanceEvent,
} from '../secure/auditLog';
import { __resetAsyncStorage } from '../../__mocks__/async-storage';
import { __resetKeychain } from '../../__mocks__/react-native-keychain';

function ev(id: string, ok = true): AttendanceEvent {
  return {
    id,
    ts: Number(id),
    ok,
    matchScore: 0.9,
    livenessScore: 0.95,
    challengePassed: true,
    deviceId: 'dev-1',
  };
}

describe('audit log', () => {
  beforeEach(async () => {
    __resetAsyncStorage();
    __resetKeychain();
    await openAuditLog();
  });

  it('appends a chained, verifiable sequence', () => {
    append(ev('1'));
    append(ev('2'));
    append(ev('3'));
    expect(pendingCount()).toBe(3);
    expect(verifyChain()).toBe(true);
  });

  it('links each event to the previous hash', () => {
    const a = append(ev('1'));
    const b = append(ev('2'));
    expect(a.prevHash).toBe('GENESIS');
    expect(b.prevHash).toBe(a.hash);
  });

  it('purges only acked ids and keeps the rest', () => {
    append(ev('1'));
    append(ev('2'));
    append(ev('3'));
    purge(['1', '2']);
    const remaining = pending().map((e) => e.id);
    expect(remaining).toEqual(['3']);
    expect(pendingCount()).toBe(1);
  });

  it('returns events oldest-first', () => {
    append(ev('10'));
    append(ev('20'));
    expect(pending().map((e) => e.id)).toEqual(['10', '20']);
  });
});
