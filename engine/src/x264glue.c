/* The engine's side of x264: open an encoder from a preset plus option strings, feed it
 * 8-bit 4:2:0 frames with forced types and quantisers, and get each encoded picture's
 * NAL units back. Kept in C so that x264_param_t and x264_picture_t, which are large and
 * change between builds, are only ever laid out by the compiler reading x264.h. */
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <x264.h>

struct rf_picture {
    int64_t pts, dts;
    int type, qp, keyframe;
};

x264_t *rf_x264_open(const char *preset, const char *tune, const char *profile, int width, int height, int fps,
                     const char *const *options, int n_options, const int region[8], const int *bands, int n_bands,
                     char *err, size_t err_len)
{
    x264_param_t p;
    if (x264_param_default_preset(&p, preset, *tune ? tune : NULL) < 0) {
        snprintf(err, err_len, "unknown x264 preset %s or tune %s", preset, tune);
        return NULL;
    }
    p.i_width = width;
    p.i_height = height;
    p.i_csp = X264_CSP_I420;
    p.i_fps_num = fps;
    p.i_fps_den = 1;
    p.i_timebase_num = 1;
    p.i_timebase_den = fps;
    p.b_vfr_input = 0;
    p.b_annexb = 0;
    p.b_repeat_headers = 0;
    p.i_log_level = X264_LOG_WARNING;
    for (int i = 0; i < n_options; i++) {
        char key[64];
        const char *eq = strchr(options[i], '=');
        size_t n = eq ? (size_t)(eq - options[i]) : strlen(options[i]);
        if (n >= sizeof key) n = sizeof key - 1;
        memcpy(key, options[i], n);
        key[n] = 0;
        if (x264_param_parse(&p, key, eq ? eq + 1 : NULL) != 0) {
            snprintf(err, err_len, "x264 rejects the option %s", options[i]);
            return NULL;
        }
    }
    p.region.i_x0 = region[0];
    p.region.i_y0 = region[1];
    p.region.i_x1 = region[2];
    p.region.i_y1 = region[3];
    p.region.b_only = region[4];
    p.region.i_mb_offset = region[5];
    p.region.b_top_closed = region[6];
    p.region.b_bottom_closed = region[7];
    if( n_bands < 0 || n_bands > (int)(sizeof p.region.i_band_y / sizeof *p.region.i_band_y) ) {
        snprintf(err, err_len, "%d bands, more than x264 takes", n_bands);
        return NULL;
    }
    p.region.i_bands = n_bands;
    for (int i = 0; i < n_bands; i++)
        p.region.i_band_y[i] = bands[i];
    if (x264_param_apply_profile(&p, profile) < 0) {
        snprintf(err, err_len, "the settings do not fit x264 profile %s", profile);
        return NULL;
    }
    x264_t *h = x264_encoder_open(&p);
    if (!h)
        snprintf(err, err_len, "x264 refused the settings (its log above says why)");
    return h;
}

int rf_x264_headers(x264_t *h, x264_nal_t **nals)
{
    int n;
    return x264_encoder_headers(h, nals, &n) < 0 ? -1 : n;
}

/* planes == NULL drains one delayed picture. quant_offsets, if not NULL, holds a qp offset
 * for every macroblock, added to qp (with aq-strength=0, it is the whole of the offset). Returns the picture's size in bytes, 0 when
 * none came out, below 0 on failure. */
int rf_x264_encode(x264_t *h, uint8_t *const planes[3], const int strides[3], int64_t pts, int type, int qp,
                   int idr_pic_id, int zones, float *quant_offsets, int32_t *copy_index, x264_copy_mb_t *copy_pool,
                   x264_nal_t **nals, int *n_nals, struct rf_picture *out)
{
    x264_picture_t in, pic;
    x264_picture_init(&in);
    if (planes) {
        in.img.i_csp = X264_CSP_I420;
        in.img.i_plane = 3;
        for (int i = 0; i < 3; i++) {
            in.img.plane[i] = planes[i];
            in.img.i_stride[i] = strides[i];
        }
        in.i_pts = pts;
        in.i_type = type;
        in.i_qpplus1 = qp >= 0 ? qp + 1 : X264_QP_AUTO;
        in.i_idr_pic_id = idr_pic_id;
        in.i_region_zones = zones;
        /* Read while the frame is taken in, so it need not outlive this call. */
        in.prop.quant_offsets = quant_offsets;
        /* Read when the frame is encoded: the caller keeps them until it comes out. */
        in.copy_index = copy_index;
        in.copy_pool = copy_pool;
    }
    int bytes = x264_encoder_encode(h, nals, n_nals, planes ? &in : NULL, &pic);
    if (bytes > 0) {
        out->pts = pic.i_pts;
        out->dts = pic.i_dts;
        out->type = pic.i_type;
        out->qp = pic.i_qpplus1 - 1;
        out->keyframe = pic.b_keyframe;
    }
    return bytes;
}

int rf_x264_copy_mb_size(void)
{
    return sizeof(x264_copy_mb_t);
}

int rf_x264_delayed(x264_t *h)
{
    return x264_encoder_delayed_frames(h);
}

int rf_x264_max_delayed(x264_t *h)
{
    return x264_encoder_maximum_delayed_frames(h);
}

void rf_x264_close(x264_t *h)
{
    x264_encoder_close(h);
}
