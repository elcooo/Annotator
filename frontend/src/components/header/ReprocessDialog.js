import React, { useEffect, useState, useContext } from 'react'
import styles from './reprocess.module.css'
import { UserContext } from '../../context/UserContext'

export default function ReprocessDialog({ token, onClose }) {
    const [_,] = useContext(UserContext) || [token]
    const authToken = token || _
    const [audioList, setAudioList] = useState([])
    const [selected, setSelected] = useState({})
    const [downsample, setDownsample] = useState(true)
    const [denoise, setDenoise] = useState(false)
    const [segment, setSegment] = useState(true)
    const [predict, setPredict] = useState(false)
    const [threshold, setThreshold] = useState(4.5)
    const [submitting, setSubmitting] = useState(false)
    const [result, setResult] = useState(null)

    useEffect(() => {
        const fetchAudio = async () => {
            const requestOptions = {
                method: 'GET',
                headers: { 'Content-Type': 'application/json', Authorization: 'Bearer ' + authToken }
            }
            const res = await fetch('http://localhost:8000/api/get-audio-files/All', requestOptions)
            if (res.ok) {
                const data = await res.json()
                setAudioList(data || [])
                const defaults = {}
                ;(data || []).forEach(a => { defaults[a.filename] = false })
                setSelected(defaults)
            }
        }
        fetchAudio()
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [])

    const toggle = (filename) => {
        setSelected(prev => ({ ...prev, [filename]: !prev[filename] }))
    }

    const submit = async () => {
        setSubmitting(true)
        setResult(null)
        const filenames = Object.keys(selected).filter(k => selected[k])
        if (filenames.length === 0) { setSubmitting(false); return }
        const requestOptions = {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', Authorization: 'Bearer ' + authToken },
            body: JSON.stringify({ filenames, downsample, denoise, segment, predict, threshold })
        }
        const res = await fetch('http://localhost:8000/api/reprocess-audio', requestOptions)
        if (res.ok) {
            const data = await res.json()
            setResult(data)
            onClose && onClose()
            window.location.reload(true)
        } else {
            setResult({ error: 'Request failed' })
        }
        setSubmitting(false)
    }

    return (
        <div className={styles.overlay}>
            <div className={styles.dialog}>
                <div className={styles.header}>
                    <h3>Reprocess Audio</h3>
                    <button className={styles.close} onClick={onClose}>×</button>
                </div>
                <div className={styles.body}>
                    <div className={styles.warning}>
                        Note: Reprocessing will delete all existing segments (including labeled ones) for the selected audio files and regenerate them.
                    </div>
                    <div className={styles.columns}>
                        <div className={styles.left}>
                            <h4>Select audio</h4>
                            <div className={styles.list}>
                                {audioList.map(a => (
                                    <label key={a.filename} className={styles.item}>
                                        <input type='checkbox' checked={!!selected[a.filename]} onChange={() => toggle(a.filename)} />
                                        <span>{a.filename}</span>
                                    </label>
                                ))}
                            </div>
                        </div>
                        <div className={styles.right}>
                            <h4>Options</h4>
                            <label className={styles.option}><input type='checkbox' checked={downsample} onChange={() => setDownsample(!downsample)} />Downsample</label>
                            <label className={styles.option}><input type='checkbox' checked={denoise} onChange={() => setDenoise(!denoise)} />Denoise</label>
                            <label className={styles.option}><input type='checkbox' checked={segment} onChange={() => setSegment(!segment)} />Segment</label>
                            <label className={styles.option}><input type='checkbox' checked={predict} onChange={() => setPredict(!predict)} />Predict</label>
                            <div className={styles.threshold}>
                                <label>Threshold: {threshold}</label>
                                <input type='range' min='1' max='9' step='0.1' value={threshold} onChange={(e) => setThreshold(parseFloat(e.target.value))} />
                            </div>
                        </div>
                    </div>
                    <div className={styles.actions}>
                        <button disabled={submitting} onClick={submit}>{submitting ? 'Processing…' : 'Run'}</button>
                    </div>
                    {result ? (
                        <div className={styles.results}>
                            {(result.results || []).map(r => (
                                <div key={r.filename} className={r.status === 'ok' ? styles.ok : styles.err}>
                                    {r.filename}: {r.status}{r.detail ? ` - ${r.detail}` : ''}
                                </div>
                            ))}
                            {result.error ? <div className={styles.err}>{result.error}</div> : null}
                        </div>
                    ) : null}
                </div>
            </div>
        </div>
    )
}


