/* The engine's side of libavcodec's H.264 decoder: the stream's own pictures, as the
 * engine's x264 wrote them (NAL units behind 4-byte lengths), back to 8-bit 4:2:0 frames.
 * Used to draw players over a recorded part of the orbit rather than render it again.
 * Kept in C, as x264glue.c is, so that libavcodec's structures are only ever laid out by
 * the compiler reading its headers. */
#include <libavcodec/avcodec.h>
#include <libavutil/frame.h>
#include <libavutil/video_enc_params.h>
#include <libavutil/motion_vector.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

struct rf_dec {
    AVCodecContext *ctx;
    AVPacket *pkt;
    AVFrame *frame;
};

/* sps and pps without lengths or start codes. NULL on failure. With `partial`, pictures
 * may lack slices: what they lack is left as it was rather than concealed, which only
 * costs time when the caller never looks there. With `vectors`, the decoder also exports
 * every frame's motion vectors (a list of all of them, built for each frame). */
struct rf_dec *rf_dec_open(const uint8_t *sps, int sps_len, const uint8_t *pps, int pps_len, int partial, int vectors)
{
    const AVCodec *codec = avcodec_find_decoder(AV_CODEC_ID_H264);
    if (!codec)
        return NULL;
    struct rf_dec *d = calloc(1, sizeof *d);
    if (!d)
        return NULL;
    d->ctx = avcodec_alloc_context3(codec);
    d->pkt = av_packet_alloc();
    d->frame = av_frame_alloc();
    if (!d->ctx || !d->pkt || !d->frame)
        goto fail;
    /* The avcC record, as flv.rs writes it: lengths of 4 bytes, one SPS, one PPS. */
    int n = 11 + sps_len + pps_len;
    uint8_t *x = av_mallocz(n + AV_INPUT_BUFFER_PADDING_SIZE);
    if (!x)
        goto fail;
    uint8_t head[6] = {1, sps[1], sps[2], sps[3], 0xff, 0xe1};
    memcpy(x, head, 6);
    x[6] = sps_len >> 8;
    x[7] = sps_len & 255;
    memcpy(x + 8, sps, sps_len);
    x[8 + sps_len] = 1;
    x[9 + sps_len] = pps_len >> 8;
    x[10 + sps_len] = pps_len & 255;
    memcpy(x + 11 + sps_len, pps, pps_len);
    d->ctx->extradata = x;
    d->ctx->extradata_size = n;
    d->ctx->thread_count = 1;
    d->ctx->flags2 |= AV_CODEC_FLAG2_FAST;
    d->ctx->export_side_data |= AV_CODEC_EXPORT_DATA_VIDEO_ENC_PARAMS;
    if (vectors)
        d->ctx->export_side_data |= AV_CODEC_EXPORT_DATA_MVS;
    if (partial) {
        d->ctx->error_concealment = 0;
        d->ctx->flags |= AV_CODEC_FLAG_OUTPUT_CORRUPT;
    }
    if (avcodec_open2(d->ctx, codec, NULL) < 0)
        goto fail;
    return d;
fail:
    av_frame_free(&d->frame);
    av_packet_free(&d->pkt);
    avcodec_free_context(&d->ctx);
    free(d);
    return NULL;
}

/* One picture in, or NULL to drain. 0 on success. */
int rf_dec_send(struct rf_dec *d, const uint8_t *data, int len, int64_t pts)
{
    if (!data)
        return avcodec_send_packet(d->ctx, NULL) < 0 ? -1 : 0;
    if (av_new_packet(d->pkt, len) < 0)
        return -1;
    memcpy(d->pkt->data, data, len);
    d->pkt->pts = pts;
    d->pkt->dts = AV_NOPTS_VALUE;
    int r = avcodec_send_packet(d->ctx, d->pkt);
    av_packet_unref(d->pkt);
    return r < 0 ? -1 : 0;
}

/* The quantiser of each of the frame's mbs macroblocks (raster order) into qp, 255 where
 * the decoder does not say. */
static void quantisers(const AVFrame *f, uint8_t *qp, int mbs)
{
    memset(qp, 255, mbs);
    const AVFrameSideData *sd = av_frame_get_side_data(f, AV_FRAME_DATA_VIDEO_ENC_PARAMS);
    if (!sd)
        return;
    AVVideoEncParams *par = (AVVideoEncParams *)sd->data;
    if (par->type != AV_VIDEO_ENC_PARAMS_H264)
        return;
    int n = par->nb_blocks < (unsigned)mbs ? (int)par->nb_blocks : mbs;
    for (int i = 0; i < n; i++) {
        int q = par->qp + av_video_enc_params_block(par, i)->delta_qp;
        qp[i] = q < 0 ? 0 : q > 51 ? 51 : q;
    }
}

/* The motion vectors of each 8x8 block of the frame's mbs macroblocks (mb_width of them
 * to a row): mv[((mb * 4 + b8) * 2 + list) * 2 + xy] in quarter pixels, b8 in raster order
 * within the macroblock, and use[(mb * 4 + b8) * 2 + list] 1 where the list predicts the
 * block, 0 where it does not, 2 where the block's parts move apart (sub-8x8 partitions).
 * Intra blocks use neither list. */
static void vectors(const AVFrame *f, int16_t *mv, int8_t *use, int mb_width, int mbs)
{
    memset(use, 0, (size_t)mbs * 8);
    const AVFrameSideData *sd = av_frame_get_side_data(f, AV_FRAME_DATA_MOTION_VECTORS);
    if (!sd)
        return;
    const AVMotionVector *v = (const AVMotionVector *)sd->data;
    int n = sd->size / sizeof *v;
    for (int i = 0; i < n; i++) {
        if (!v[i].motion_scale)
            continue;
        int list = v[i].source > 0;
        int mx = v[i].motion_x * 4 / v[i].motion_scale, my = v[i].motion_y * 4 / v[i].motion_scale;
        int x0 = v[i].dst_x - v[i].w / 2, y0 = v[i].dst_y - v[i].h / 2;
        for (int y = y0; y < y0 + v[i].h; y += 8)
            for (int x = x0; x < x0 + v[i].w; x += 8) {
                if (x < 0 || y < 0)
                    continue;
                int mb = (y / 16) * mb_width + x / 16;
                if (x / 16 >= mb_width || mb >= mbs)
                    continue;
                int b8 = ((y / 8) & 1) * 2 + ((x / 8) & 1);
                int k = (mb * 4 + b8) * 2 + list;
                int same = mv[k * 2] == mx && mv[k * 2 + 1] == my;
                if (use[k] == 0) {
                    mv[k * 2] = mx;
                    mv[k * 2 + 1] = my;
                    use[k] = (v[i].w < 8 || v[i].h < 8) ? 2 : 1;
                } else if (!same) {
                    use[k] = 2;
                }
            }
    }
}

/* The next frame out, in display order, copied into yuv (width * height * 3 / 2, planar
 * 4:2:0), only the rows of the macroblock rows whose byte in `rows` is not 0 if rows is
 * not NULL; its pts in *pts and, if qp is not NULL, its macroblocks' quantisers there (one
 * byte for each of the frame's macroblocks), and if mv is not NULL, their motion vectors
 * as `vectors` gives them. 1 when a frame came out, 0 when the decoder needs more, -1 on
 * failure. */
int rf_dec_receive(struct rf_dec *d, uint8_t *yuv, const uint8_t *rows, uint8_t *qp, int16_t *mv, int8_t *mv_use, int width,
                   int height, int64_t *pts)
{
    int r = avcodec_receive_frame(d->ctx, d->frame);
    if (r == AVERROR(EAGAIN) || r == AVERROR_EOF)
        return 0;
    if (r < 0)
        return -1;
    AVFrame *f = d->frame;
    int ok = f->format == AV_PIX_FMT_YUV420P && f->width == width && f->height == height;
    if (ok) {
        uint8_t *dst = yuv;
        for (int p = 0; p < 3; p++) {
            int w = p ? width / 2 : width, h = p ? height / 2 : height, mb = p ? 8 : 16;
            for (int y = 0; y < h; y++)
                if (!rows || rows[y / mb])
                    memcpy(dst + (size_t)y * w, f->data[p] + (size_t)y * f->linesize[p], w);
            dst += w * h;
        }
        *pts = f->pts;
        if (qp)
            quantisers(f, qp, ((width + 15) / 16) * ((height + 15) / 16));
        if (mv)
            vectors(f, mv, mv_use, (width + 15) / 16, ((width + 15) / 16) * ((height + 15) / 16));
    }
    av_frame_unref(f);
    return ok ? 1 : -1;
}

/* Forgets every picture held, ready for the next IDR. */
void rf_dec_reset(struct rf_dec *d)
{
    avcodec_flush_buffers(d->ctx);
}

void rf_dec_close(struct rf_dec *d)
{
    if (!d)
        return;
    av_frame_free(&d->frame);
    av_packet_free(&d->pkt);
    avcodec_free_context(&d->ctx);
    free(d);
}
