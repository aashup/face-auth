import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  ActivityIndicator,
  Alert,
  InteractionManager,
  Modal,
  SafeAreaView,
  StyleSheet,
  Text,
  TextInput,
  TouchableOpacity,
  View,
} from 'react-native';
import { useTensorflowModel } from 'react-native-fast-tflite';
import {
  FaceAuth,
  FaceAuthView,
  DEFAULT_PERFORMANCE,
  type AuthMode,
  type AuthResult,
  type FaceAuthModels,
  type SyncStatus,
} from 'react-native-offline-face-auth';

/**
 * Reference host — local multi-template enrollment (no server required).
 *
 * ── Enrollment ──────────────────────────────────────────────────────────────
 * Tap "Enroll" → enter a name → camera opens ENROLL_SHOTS times in sequence,
 * capturing one face embedding per session → all embeddings stored locally via
 * FaceAuth.enrollLocal() (encrypted on-device, cancelable transform applied).
 *
 * ── Authentication ──────────────────────────────────────────────────────────
 * Tap "Authenticate" → the pipeline scores every stored template and returns
 * the best match. A popup shows the matched name and score.
 *
 * ── Configuration ───────────────────────────────────────────────────────────
 * Change ENROLL_SHOTS below to control how many face captures are collected per
 * person. 3 (default) gives the matcher more angles to work with; 1 is the
 * fastest "quick enroll" mode.
 */

// ── ✏️  Configurable: number of face captures per enrollment ─────────────────
const ENROLL_SHOTS = 3; // 1 = single-shot,  2–5 = multi-shot (recommended: 3)
// ─────────────────────────────────────────────────────────────────────────────

// ── ✏️  Optional server sync (leave undefined for fully-offline mode) ─────────
// Uncomment and fill these only if you want server-based provisioning/sync:
// const AWS_URL      = 'http://localhost:8080';
// const DEVICE_TOKEN = 'demo-device-token';
// ─────────────────────────────────────────────────────────────────────────────

export default function App() {
  const [ready, setReady]     = useState(false);
  const [last, setLast]       = useState<AuthResult | null>(null);
  const [sync, setSync]       = useState<SyncStatus | null>(null);
  const [session, setSession] = useState<AuthMode | null>(null);

  // ── Enrollment dialog state ─────────────────────────────────────────────────
  const [enrollDialogVisible, setEnrollDialogVisible] = useState(false);
  const [enrollInput, setEnrollInput]                 = useState('');

  // ── Multi-shot capture state ────────────────────────────────────────────────
  // pendingEnrollId  — name typed in the dialog, used for all shots
  // captureStep      — which shot we're on (1-based); 0 = not enrolling
  // collectedEmbs    — base64 embeddings gathered so far
  const pendingEnrollId  = useRef('');
  const captureStep      = useRef(0);
  const collectedEmbs    = useRef<string[]>([]);

  // ── Duplicate-check overlay state ──────────────────────────────────────────
  const [checkingDuplicate, setCheckingDuplicate] = useState(false);

  // ── Model loading ───────────────────────────────────────────────────────────
  // Improvement 1: per-model XNNPACK thread counts from DEFAULT_PERFORMANCE.
  // To override, pass a custom `performance` object to FaceAuth.init() and use
  // resolvePerformance() here instead of DEFAULT_PERFORMANCE.
  //
  // Thread budget rationale (DEFAULT_PERFORMANCE):
  //   blazeface / faceMesh  → 2 threads (small/medium model; XNNPACK overhead
  //                           outweighs benefit beyond 2 on most devices)
  //   liveness0 / liveness1 → 4 threads (MiniFASNet runs in parallel via JS
  //                           Promise.all; each branch gets full thread budget)
  //   facenet               → 4 threads (heaviest model — most gain from threads)
  const blazeface = useTensorflowModel(
    require('./assets/models/blazeface.tflite'),
    'xnnpack',
    { numThreads: DEFAULT_PERFORMANCE.blazefaceThreads },
  );
  const faceMesh  = useTensorflowModel(
    require('./assets/models/face_mesh.tflite'),
    'xnnpack',
    { numThreads: DEFAULT_PERFORMANCE.faceMeshThreads },
  );
  const facenet   = useTensorflowModel(
    require('./assets/models/facenet_512.tflite'),
    'xnnpack',
    { numThreads: DEFAULT_PERFORMANCE.embeddingThreads },
  );
  const spoof27   = useTensorflowModel(
    require('./assets/models/spoof_2_7.tflite'),
    'xnnpack',
    { numThreads: DEFAULT_PERFORMANCE.livenessThreads },
  );
  const spoof40   = useTensorflowModel(
    require('./assets/models/spoof_4_0.tflite'),
    'xnnpack',
    { numThreads: DEFAULT_PERFORMANCE.livenessThreads },
  );

  const loaded = [blazeface, faceMesh, facenet, spoof27, spoof40].every(
    (m) => m.state === 'loaded',
  );

  useEffect(() => {
    let unsub = () => {};
    (async () => {
      try {
        await FaceAuth.init({
          // awsSyncUrl and deviceToken are omitted → fully offline mode.
          // Uncomment + fill both to enable server provisioning / attendance sync:
          // awsSyncUrl:  AWS_URL,
          // deviceToken: DEVICE_TOKEN,
          deviceId: 'demo-device-01',
          debug:    false,
          // Thread counts must match what useTensorflowModel was called with above.
          // Stored in config so useFrameSource can read frameSize automatically.
          performance: DEFAULT_PERFORMANCE,
        });
        unsub = FaceAuth.onSyncStatus(setSync);
        setReady(true);
      } catch (e: any) {
        console.error('FaceAuth.init failed:', e?.message ?? String(e));
      }
    })();
    return () => { unsub(); void FaceAuth.dispose(); };
  }, []);

  const models: FaceAuthModels | null = useMemo(
    () =>
      loaded
        ? {
            blazeface:  blazeface.model as any,
            faceMesh:   faceMesh.model as any,
            embedding:  facenet.model as any,
            liveness0:  spoof27.model as any,
            liveness1:  spoof40.model as any,
          }
        : null,
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [loaded],
  );

  // ── Start first (or next) enroll shot ───────────────────────────────────────
  const startNextShot = () => {
    captureStep.current += 1;
    setSession('enroll');
  };

  /**
   * Run a duplicate-enrollment check off the critical path.
   *
   * `InteractionManager.runAfterInteractions` defers the heavy synchronous
   * dot-product scan until any pending UI animations (modal close, state
   * transitions) have flushed — so the camera close animation plays smoothly
   * before the scan blocks the JS thread.
   *
   * Returns a Promise so the caller can `await` it and decide whether to
   * continue to the next shot or abort.
   */
  const runDuplicateCheck = useCallback((embeddingB64: string): Promise<boolean> => {
    return new Promise((resolve) => {
      InteractionManager.runAfterInteractions(async () => {
        setCheckingDuplicate(true);
        try {
          const result = await FaceAuth.checkDuplicate(embeddingB64);
          if (result.isDuplicate) {
            // Reset enrollment state before showing alert so the user
            // can start fresh without stale data.
            captureStep.current   = 0;
            collectedEmbs.current = [];
            Alert.alert(
              '⚠️  Already Registered',
              `This face is already enrolled as:\n\n` +
              `"${result.personnelId}"\n` +
              `Score: ${result.score.toFixed(2)}\n\n` +
              `Please use that name to authenticate.`,
              [{ text: 'OK', style: 'cancel' }],
            );
            resolve(false); // duplicate found → abort enrollment
          } else {
            resolve(true); // no duplicate → safe to continue
          }
        } catch (e: any) {
          // On error, be permissive — let enrollment proceed; don't block
          // the user on a check that should be best-effort.
          console.warn('[checkDuplicate] error:', e?.message ?? String(e));
          resolve(true);
        } finally {
          setCheckingDuplicate(false);
        }
      });
    });
  }, []);

  // ── Handle every FaceAuthView result ───────────────────────────────────────
  const handleResult = async (r: AuthResult, activeSession: AuthMode) => {
    setSession(null);

    // ── Enroll shot completed ─────────────────────────────────────────────────
    if (activeSession === 'enroll') {
      if (!r.ok || !r.embeddingB64) {
        // Capture failed (liveness / challenge / cancelled) — restart this shot
        Alert.alert(
          `Shot ${captureStep.current} failed`,
          `Reason: ${r.failureReason ?? 'unknown'}. Try again.`,
          [
            { text: 'Retry', onPress: () => setSession('enroll') },
            {
              text: 'Cancel',
              style: 'cancel',
              onPress: () => {
                captureStep.current = 0;
                collectedEmbs.current = [];
              },
            },
          ],
        );
        return;
      }

      // Shot succeeded — on the very first shot, check for duplicate enrollment
      // before collecting further shots. This avoids wasting the user's time on
      // 2–4 more captures only to reject them at the final step.
      if (captureStep.current === 1) {
        const ok = await runDuplicateCheck(r.embeddingB64);
        if (!ok) return; // duplicate detected — alert already shown, state reset
      }

      // Accumulate embedding
      collectedEmbs.current.push(r.embeddingB64);

      if (captureStep.current < ENROLL_SHOTS) {
        // More shots needed — open camera again after a short prompt
        Alert.alert(
          `Shot ${captureStep.current} of ${ENROLL_SHOTS} captured ✓`,
          `${ENROLL_SHOTS - captureStep.current} more shot(s) needed. Slightly change your angle or expression.`,
          [{ text: 'Next shot', onPress: startNextShot }],
        );
      } else {
        // All shots done — store all embeddings locally
        try {
          await FaceAuth.enrollLocal(pendingEnrollId.current, collectedEmbs.current);
          Alert.alert(
            'Enrolled ✓',
            `"${pendingEnrollId.current}" stored with ${collectedEmbs.current.length} face template(s).`,
          );
        } catch (e: any) {
          Alert.alert('Enroll failed', e?.message ?? 'unknown error');
        } finally {
          captureStep.current = 0;
          collectedEmbs.current = [];
        }
      }
      return;
    }

    // ── Identify / Verify result ──────────────────────────────────────────────
    if (activeSession === 'identify' || activeSession === 'verify') {
      if (r.ok && r.personnelId) {
        Alert.alert(
          '✅ Match found',
          `Person: ${r.personnelId}\nScore: ${r.matchScore.toFixed(2)}`,
        );
      } else {
        Alert.alert('❌ No match', `Reason: ${r.failureReason ?? 'unknown'}`);
      }
    }

    setLast(r);
  };

  // ── Loading screen ──────────────────────────────────────────────────────────
  if (!ready || !models) {
    return (
      <SafeAreaView style={styles.center}>
        <Text style={styles.text}>Loading models…</Text>
      </SafeAreaView>
    );
  }

  // ── Camera session ──────────────────────────────────────────────────────────
  if (session) {
    const isEnroll = session === 'enroll';

    return (
      <View style={{ flex: 1 }}>
        <FaceAuthView
          mode={session}
          models={models}
          onClose={() => {
            if (isEnroll) {
              captureStep.current   = 0;
              collectedEmbs.current = [];
            }
            setSession(null);
          }}
          onResult={(r) => void handleResult(r, session)}
        />

        {/* Shot counter badge — shown on top of the camera during enrollment */}
        {isEnroll && (
          <View style={styles.shotBadge} pointerEvents="none">
            <Text style={styles.shotBadgeStep}>
              Shot {captureStep.current} of {ENROLL_SHOTS}
            </Text>
            <Text style={styles.shotBadgeName}>
              {pendingEnrollId.current}
            </Text>
          </View>
        )}
      </View>
    );
  }

  // ── Duplicate-checking overlay (shown after first shot while scan runs) ────
  if (checkingDuplicate) {
    return (
      <SafeAreaView style={styles.center}>
        <View style={styles.checkingCard}>
          <ActivityIndicator size="large" color="#0ea5e9" />
          <Text style={styles.checkingText}>Checking for duplicates…</Text>
          <Text style={styles.checkingSub}>
            Making sure this face isn't already enrolled
          </Text>
        </View>
      </SafeAreaView>
    );
  }

  // ── Main screen ─────────────────────────────────────────────────────────────
  return (
    <SafeAreaView style={styles.container}>

      {/* ── Enrollment name dialog ────────────────────────────────────────── */}
      <Modal
        visible={enrollDialogVisible}
        transparent
        animationType="fade"
        onRequestClose={() => setEnrollDialogVisible(false)}>
        <View style={styles.overlay}>
          <View style={styles.dialog}>
            <Text style={styles.dialogTitle}>Enter Person ID / Name</Text>
            <Text style={styles.dialogSub}>
              {ENROLL_SHOTS === 1
                ? '1 face capture will be taken.'
                : `${ENROLL_SHOTS} face captures will be taken for better accuracy.`}
            </Text>
            <TextInput
              style={styles.input}
              placeholder="e.g. John Smith or emp-002"
              placeholderTextColor="#94a3b8"
              value={enrollInput}
              onChangeText={setEnrollInput}
              autoFocus
              returnKeyType="done"
              onSubmitEditing={() => {
                const id = enrollInput.trim();
                if (!id) return;
                setEnrollDialogVisible(false);
                pendingEnrollId.current  = id;
                captureStep.current      = 0;
                collectedEmbs.current    = [];
                startNextShot();
              }}
            />
            <View style={styles.dialogBtns}>
              <TouchableOpacity
                style={[styles.dialogBtn, styles.dialogBtnCancel]}
                onPress={() => setEnrollDialogVisible(false)}>
                <Text style={styles.dialogBtnText}>Cancel</Text>
              </TouchableOpacity>
              <TouchableOpacity
                style={[styles.dialogBtn, styles.dialogBtnOk]}
                onPress={() => {
                  const id = enrollInput.trim();
                  if (!id) return;
                  setEnrollDialogVisible(false);
                  pendingEnrollId.current  = id;
                  captureStep.current      = 0;
                  collectedEmbs.current    = [];
                  startNextShot();
                }}>
                <Text style={styles.dialogBtnText}>Start Enroll</Text>
              </TouchableOpacity>
            </View>
          </View>
        </View>
      </Modal>

      <Text style={styles.title}>Offline Face Auth — Demo</Text>

      {/* 1. Enroll */}
      <TouchableOpacity
        style={styles.btn}
        onPress={() => { setEnrollInput(''); setEnrollDialogVisible(true); }}>
        <Text style={styles.btnText}>
          1. Enroll my face ({ENROLL_SHOTS} shot{ENROLL_SHOTS !== 1 ? 's' : ''}, local)
        </Text>
      </TouchableOpacity>

      {/* 2. Provision (server-based, optional) */}
      <TouchableOpacity
        style={[styles.btn, styles.btnSecondary]}
        onPress={async () => {
          const r = await FaceAuth.provision();
          setLast(null);
          Alert.alert(
            r.ok ? `Provisioned ${r.count} templates` : 'Provision failed',
            r.error ?? '',
          );
        }}>
        <Text style={styles.btnText}>2. Provision templates (server, optional)</Text>
      </TouchableOpacity>

      {/* 3. Authenticate */}
      <TouchableOpacity style={styles.btn} onPress={() => setSession('identify')}>
        <Text style={styles.btnText}>3. Authenticate (offline)</Text>
      </TouchableOpacity>

      {/* 4. Sync attendance */}
      <TouchableOpacity style={[styles.btn, styles.btnSecondary]} onPress={() => FaceAuth.syncNow()}>
        <Text style={styles.btnText}>4. Sync attendance to server</Text>
      </TouchableOpacity>

      {/* Status panel */}
      <View style={styles.panel}>
        <Text style={styles.label}>
          Templates on device: {FaceAuth.getTemplateCount()}
        </Text>
        <Text style={styles.label}>Pending events: {sync?.pendingCount ?? 0}</Text>
        <Text style={styles.label}>Sync: {sync?.state ?? 'idle'}</Text>
        {last && (
          <Text style={styles.label}>
            Last:{' '}
            {last.ok
              ? `✅ ${last.personnelId ?? 'enrolled'}`
              : `❌ ${last.failureReason}`}
            {' · '}match {last.matchScore.toFixed(2)} · live{' '}
            {last.livenessScore.toFixed(2)}
          </Text>
        )}
      </View>

      {/* Hint about ENROLL_SHOTS */}
      <Text style={styles.hint}>
        Change <Text style={styles.hintCode}>ENROLL_SHOTS</Text> at the top of App.tsx to
        adjust how many captures are taken per person (default: 3).
      </Text>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container:       { flex: 1, padding: 24, gap: 12, backgroundColor: '#0f172a' },
  center:          { flex: 1, alignItems: 'center', justifyContent: 'center', backgroundColor: '#0f172a' },
  title:           { color: '#fff', fontSize: 22, fontWeight: '700', marginBottom: 8 },
  btn:             { backgroundColor: '#0ea5e9', padding: 16, borderRadius: 10 },
  btnSecondary:    { backgroundColor: '#334155' },
  btnText:         { color: '#fff', fontSize: 15, fontWeight: '600', textAlign: 'center' },
  panel:           { marginTop: 16, gap: 4, backgroundColor: '#1e293b', padding: 14, borderRadius: 10 },
  label:           { color: '#e2e8f0', fontSize: 14 },
  hint:            { color: '#64748b', fontSize: 12, marginTop: 8, textAlign: 'center' },
  hintCode:        { color: '#38bdf8', fontFamily: 'monospace' },
  // ── Shot counter badge (enroll mode, sits above the camera) ─────────────────
  shotBadge: {
    position: 'absolute',
    top: 52,
    left: 0,
    right: 0,
    alignItems: 'center',
  },
  shotBadgeStep: {
    color: '#fff',
    fontSize: 18,
    fontWeight: '800',
    backgroundColor: 'rgba(14,165,233,0.92)',
    paddingHorizontal: 22,
    paddingVertical: 8,
    borderRadius: 24,
    overflow: 'hidden',
    textAlign: 'center',
  },
  shotBadgeName: {
    color: 'rgba(255,255,255,0.85)',
    fontSize: 13,
    marginTop: 6,
    backgroundColor: 'rgba(0,0,0,0.45)',
    paddingHorizontal: 14,
    paddingVertical: 4,
    borderRadius: 12,
    overflow: 'hidden',
  },
  // ── Dialog ───────────────────────────────────────────────────────────────────
  overlay:         { flex: 1, backgroundColor: 'rgba(0,0,0,0.65)', justifyContent: 'center', padding: 32 },
  dialog:          { backgroundColor: '#1e293b', borderRadius: 16, padding: 24, gap: 14 },
  dialogTitle:     { color: '#fff', fontSize: 18, fontWeight: '700' },
  dialogSub:       { color: '#94a3b8', fontSize: 13 },
  input:           {
    backgroundColor: '#0f172a',
    color: '#fff',
    borderRadius: 8,
    padding: 12,
    fontSize: 16,
    borderWidth: 1,
    borderColor: '#334155',
  },
  dialogBtns:      { flexDirection: 'row', gap: 12, justifyContent: 'flex-end' },
  dialogBtn:       { paddingVertical: 10, paddingHorizontal: 20, borderRadius: 8 },
  dialogBtnCancel: { backgroundColor: '#475569' },
  dialogBtnOk:     { backgroundColor: '#0ea5e9' },
  dialogBtnText:   { color: '#fff', fontWeight: '600', fontSize: 15 },
  // ── Duplicate-check overlay ───────────────────────────────────────────────
  checkingCard: {
    backgroundColor: '#1e293b',
    borderRadius: 20,
    padding: 32,
    alignItems: 'center',
    gap: 16,
    marginHorizontal: 32,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.4,
    shadowRadius: 12,
    elevation: 10,
  },
  checkingText: {
    color: '#fff',
    fontSize: 17,
    fontWeight: '700',
    textAlign: 'center',
  },
  checkingSub: {
    color: '#94a3b8',
    fontSize: 13,
    textAlign: 'center',
  },
});
