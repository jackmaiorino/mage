package mage.player.ai.rl;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.EOFException;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Minimal framed binary protocol for the shared GPU host.
 *
 * Each frame starts with an int32 body length followed by a body:
 *
 * request:
 *   int32 opcode
 *   int64 requestId
 *   map<string,string> headers
 *   bytes payload
 *
 * response:
 *   int32 status
 *   int64 requestId
 *   map<string,string> headers
 *   bytes payload
 */
public final class SharedGpuProtocol {

    public static final int OP_REGISTER_PROFILE = 1;
    public static final int OP_SCORE = 2;
    public static final int OP_ENQUEUE_TRAIN = 3;
    public static final int OP_SAVE_MODEL = 4;
    public static final int OP_GET_DEVICE_INFO = 5;
    public static final int OP_GET_MAIN_STATS = 6;
    public static final int OP_GET_MULLIGAN_STATS = 7;
    public static final int OP_GET_HEALTH_STATS = 8;
    public static final int OP_RESET_HEALTH_STATS = 9;
    public static final int OP_RECORD_GAME_RESULT = 10;
    public static final int OP_PREDICT_MULLIGAN = 11;
    public static final int OP_PREDICT_MULLIGAN_SCORES = 12;
    public static final int OP_TRAIN_MULLIGAN = 13;
    public static final int OP_SAVE_MULLIGAN_MODEL = 14;
    public static final int OP_GET_VALUE_HEAD_METRICS = 15;
    public static final int OP_CLOSE_PROFILE = 16;

    public static final int STATUS_OK = 0;
    public static final int STATUS_ERROR = 1;

    private SharedGpuProtocol() {
    }

    public static final class RequestFrame {
        public final int opcode;
        public final long requestId;
        public final Map<String, String> headers;
        public final byte[] payload;

        RequestFrame(int opcode, long requestId, Map<String, String> headers, byte[] payload) {
            this.opcode = opcode;
            this.requestId = requestId;
            this.headers = headers;
            this.payload = payload;
        }
    }

    public static final class ResponseFrame {
        public final int status;
        public final long requestId;
        public final Map<String, String> headers;
        public final byte[] payload;

        ResponseFrame(int status, long requestId, Map<String, String> headers, byte[] payload) {
            this.status = status;
            this.requestId = requestId;
            this.headers = headers;
            this.payload = payload;
        }
    }

    public static void writeRequest(OutputStream output, int opcode, long requestId, Map<String, String> headers, byte[] payload)
            throws IOException {
        byte[] body = buildFrameBody(opcode, requestId, headers, payload);
        DataOutputStream out = new DataOutputStream(output);
        out.writeInt(body.length);
        out.write(body);
        out.flush();
    }

    public static void writeResponse(OutputStream output, int status, long requestId, Map<String, String> headers, byte[] payload)
            throws IOException {
        byte[] body = buildFrameBody(status, requestId, headers, payload);
        DataOutputStream out = new DataOutputStream(output);
        out.writeInt(body.length);
        out.write(body);
        out.flush();
    }

    public static RequestFrame readRequest(InputStream input) throws IOException {
        byte[] body = readFrameBody(input);
        DataInputStream in = new DataInputStream(new ByteArrayInputStream(body));
        int opcode = in.readInt();
        long requestId = in.readLong();
        Map<String, String> headers = readHeaders(in);
        byte[] payload = readPayload(in);
        return new RequestFrame(opcode, requestId, headers, payload);
    }

    public static ResponseFrame readResponse(InputStream input) throws IOException {
        byte[] body = readFrameBody(input);
        DataInputStream in = new DataInputStream(new ByteArrayInputStream(body));
        int status = in.readInt();
        long requestId = in.readLong();
        Map<String, String> headers = readHeaders(in);
        byte[] payload = readPayload(in);
        return new ResponseFrame(status, requestId, headers, payload);
    }

    private static byte[] buildFrameBody(int code, long requestId, Map<String, String> headers, byte[] payload)
            throws IOException {
        ByteArrayOutputStream buffer = new ByteArrayOutputStream();
        DataOutputStream out = new DataOutputStream(buffer);
        out.writeInt(code);
        out.writeLong(requestId);
        writeHeaders(out, headers);
        writePayload(out, payload);
        out.flush();
        return buffer.toByteArray();
    }

    private static byte[] readFrameBody(InputStream input) throws IOException {
        DataInputStream in = new DataInputStream(input);
        int frameLength;
        try {
            frameLength = in.readInt();
        } catch (EOFException eof) {
            throw eof;
        }
        if (frameLength < 0 || frameLength > (256 * 1024 * 1024)) {
            throw new IOException("Invalid shared GPU frame length: " + frameLength);
        }
        byte[] body = new byte[frameLength];
        in.readFully(body);
        return body;
    }

    private static void writeHeaders(DataOutputStream out, Map<String, String> headers) throws IOException {
        Map<String, String> safe = headers == null ? Collections.emptyMap() : headers;
        out.writeInt(safe.size());
        for (Map.Entry<String, String> entry : safe.entrySet()) {
            writeString(out, entry.getKey());
            writeString(out, entry.getValue());
        }
    }

    private static Map<String, String> readHeaders(DataInputStream in) throws IOException {
        int count = in.readInt();
        if (count <= 0) {
            return Collections.emptyMap();
        }
        Map<String, String> headers = new LinkedHashMap<>(count);
        for (int i = 0; i < count; i++) {
            headers.put(readString(in), readString(in));
        }
        return headers;
    }

    private static void writePayload(DataOutputStream out, byte[] payload) throws IOException {
        byte[] safe = payload == null ? new byte[0] : payload;
        out.writeInt(safe.length);
        out.write(safe);
    }

    private static byte[] readPayload(DataInputStream in) throws IOException {
        int payloadLength = in.readInt();
        if (payloadLength < 0) {
            throw new IOException("Negative shared GPU payload length: " + payloadLength);
        }
        byte[] payload = new byte[payloadLength];
        in.readFully(payload);
        return payload;
    }

    private static void writeString(DataOutputStream out, String value) throws IOException {
        byte[] bytes = (value == null ? "" : value).getBytes(StandardCharsets.UTF_8);
        out.writeInt(bytes.length);
        out.write(bytes);
    }

    private static String readString(DataInputStream in) throws IOException {
        int length = in.readInt();
        if (length < 0 || length > (8 * 1024 * 1024)) {
            throw new IOException("Invalid shared GPU string length: " + length);
        }
        byte[] bytes = new byte[length];
        in.readFully(bytes);
        return new String(bytes, StandardCharsets.UTF_8);
    }
}
