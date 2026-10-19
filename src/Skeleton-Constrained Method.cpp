#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#endif

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <queue>
#include <set>
#include <algorithm>
#include <functional>
#include <cmath>
#include <climits>
#include <cfloat>
#include <chrono>
#include <iomanip>
#include <fstream>
#include <filesystem>
#include <sstream>
#include <cctype>
#include <stdexcept>
#include "json.hpp"

#ifdef _WIN32
#include <windows.h>
#endif

#ifdef _MSC_VER
#pragma execution_character_set("utf-8")
#endif

using json = nlohmann::json;
namespace fs = std::filesystem;

using namespace cv;
using namespace std;

static const int BIN_THRESH = 200;                   // Grayscale binarization threshold
static const int MIN_AREA = 4000;                    // Minimum connected-component area (pixels)
static const int PEAK_MIN_DIST = 60;                 // NMS minimum distance between centers (pixels)
static const int CONCAVE_STEP = 7;                   // Contour sampling step for concave-point detection
static const double CONCAVE_ANGLE_THRESH = 140.0;   // Concave-point detection angle threshold (degrees)
static const int CONCAVE_GROUP_GAP = 3;             // Merge neighboring concave points when index gap < 3

struct CenterPeakStats
{
    long long rawPeakCount = 0;
    long long usedCenterCount = 0;
};

static const int dx8[8] = { -1, 0, 1, -1, 1, -1, 0, 1 };
static const int dy8[8] = { -1,-1,-1,  0, 0,  1, 1, 1 };

inline bool insideImage(int x, int y, int cols, int rows)
{
    return x >= 0 && x < cols && y >= 0 && y < rows;
}

Mat fillSmallInternalHoles(const Mat& foregroundMask, int maxHoleArea)
{
    CV_Assert(foregroundMask.type() == CV_8U);

    Mat inv;
    bitwise_not(foregroundMask, inv); 

    Mat labels, stats, centroids;
    int nLabels = connectedComponentsWithStats(inv, labels, stats, centroids, 8);
    if (nLabels <= 1) return foregroundMask.clone();

    vector<uchar> touchesBorder(nLabels, 0);

    for (int x = 0; x < labels.cols; ++x)
    {
        touchesBorder[labels.at<int>(0, x)] = 1;
        touchesBorder[labels.at<int>(labels.rows - 1, x)] = 1;
    }
    for (int y = 0; y < labels.rows; ++y)
    {
        touchesBorder[labels.at<int>(y, 0)] = 1;
        touchesBorder[labels.at<int>(y, labels.cols - 1)] = 1;
    }

    vector<uchar> fillLabel(nLabels, 0);
    for (int lb = 1; lb < nLabels; ++lb)
    {
        int area = stats.at<int>(lb, CC_STAT_AREA);
        if (!touchesBorder[lb] && area < maxHoleArea)
        {
            fillLabel[lb] = 1;
        }
    }

    Mat filled = foregroundMask.clone();
    for (int y = 0; y < labels.rows; ++y)
    {
        const int* pl = labels.ptr<int>(y);
        uchar* pf = filled.ptr<uchar>(y);
        for (int x = 0; x < labels.cols; ++x)
        {
            if (fillLabel[pl[x]])
                pf[x] = 255;
        }
    }

    return filled;
}

void thinningZhangSuen(Mat& im)
{
    Mat img = im.clone();
    img /= 255;

    vector<Point> delList;
    delList.reserve(img.rows * img.cols / 10);

    while (true)
    {
        delList.clear();

        for (int y = 1; y < img.rows - 1; ++y)
        {
            uchar* row = img.ptr<uchar>(y);
            for (int x = 1; x < img.cols - 1; ++x)
            {
                if (row[x] != 1) continue;

                int p2 = img.at<uchar>(y - 1, x);
                int p3 = img.at<uchar>(y - 1, x + 1);
                int p4 = img.at<uchar>(y, x + 1);
                int p5 = img.at<uchar>(y + 1, x + 1);
                int p6 = img.at<uchar>(y + 1, x);
                int p7 = img.at<uchar>(y + 1, x - 1);
                int p8 = img.at<uchar>(y, x - 1);
                int p9 = img.at<uchar>(y - 1, x - 1);

                int A = (p2 == 0 && p3 == 1) + (p3 == 0 && p4 == 1) +
                    (p4 == 0 && p5 == 1) + (p5 == 0 && p6 == 1) +
                    (p6 == 0 && p7 == 1) + (p7 == 0 && p8 == 1) +
                    (p8 == 0 && p9 == 1) + (p9 == 0 && p2 == 1);
                int B = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9;

                if (A == 1 && B >= 2 && B <= 6 &&
                    (p2 * p4 * p6 == 0) && (p4 * p6 * p8 == 0))
                {
                    delList.push_back(Point(x, y));
                }
            }
        }

        for (const Point& p : delList) img.at<uchar>(p.y, p.x) = 0;
        if (delList.empty()) break;

        delList.clear();

        for (int y = 1; y < img.rows - 1; ++y)
        {
            uchar* row = img.ptr<uchar>(y);
            for (int x = 1; x < img.cols - 1; ++x)
            {
                if (row[x] != 1) continue;

                int p2 = img.at<uchar>(y - 1, x);
                int p3 = img.at<uchar>(y - 1, x + 1);
                int p4 = img.at<uchar>(y, x + 1);
                int p5 = img.at<uchar>(y + 1, x + 1);
                int p6 = img.at<uchar>(y + 1, x);
                int p7 = img.at<uchar>(y + 1, x - 1);
                int p8 = img.at<uchar>(y, x - 1);
                int p9 = img.at<uchar>(y - 1, x - 1);

                int A = (p2 == 0 && p3 == 1) + (p3 == 0 && p4 == 1) +
                    (p4 == 0 && p5 == 1) + (p5 == 0 && p6 == 1) +
                    (p6 == 0 && p7 == 1) + (p7 == 0 && p8 == 1) +
                    (p8 == 0 && p9 == 1) + (p9 == 0 && p2 == 1);
                int B = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9;

                if (A == 1 && B >= 2 && B <= 6 &&
                    (p2 * p4 * p8 == 0) && (p2 * p6 * p8 == 0))
                {
                    delList.push_back(Point(x, y));
                }
            }
        }

        for (const Point& p : delList) img.at<uchar>(p.y, p.x) = 0;
        if (delList.empty()) break;
    }

    im = img * 255;
}

vector<Point> detectLocalMaximaPerBlobFast(const Mat& dist, int* rawPeakCount = nullptr)
{
    Mat distDilated;
    dilate(dist, distDilated, Mat());

    Mat peakMask = Mat::zeros(dist.size(), CV_8U);
    for (int y = 0; y < dist.rows; ++y)
    {
        const float* pd = dist.ptr<float>(y);
        const float* pdd = distDilated.ptr<float>(y);
        uchar* pm = peakMask.ptr<uchar>(y);
        for (int x = 0; x < dist.cols; ++x)
        {
            if (pd[x] > 1e-6f && fabs(pd[x] - pdd[x]) < 1e-6f)
                pm[x] = 255;
        }
    }

    Mat labels, stats, centroids;
    int nLabels = connectedComponentsWithStats(peakMask, labels, stats, centroids, 8);
    if (nLabels <= 1) return {};

    vector<float> bestVal(nLabels, -1.0f);
    vector<Point> bestPt(nLabels, Point(-1, -1));

    for (int y = 0; y < labels.rows; ++y)
    {
        const int* pl = labels.ptr<int>(y);
        const float* pd = dist.ptr<float>(y);
        for (int x = 0; x < labels.cols; ++x)
        {
            int lb = pl[x];
            if (lb <= 0) continue;
            if (pd[x] > bestVal[lb])
            {
                bestVal[lb] = pd[x];
                bestPt[lb] = Point(x, y);
            }
        }
    }

    vector<Point> allPeaks;
    for (int lb = 1; lb < nLabels; ++lb)
    {
        if (bestPt[lb].x >= 0) allPeaks.push_back(bestPt[lb]);
    }

    stable_sort(allPeaks.begin(), allPeaks.end(),
        [&](const Point& a, const Point& b)
        {
            return dist.at<float>(a.y, a.x) > dist.at<float>(b.y, b.x);
        });

    if (rawPeakCount) *rawPeakCount = (int)allPeaks.size();

    const int minDist2 = PEAK_MIN_DIST * PEAK_MIN_DIST;
    vector<Point> filtered;
    filtered.reserve(allPeaks.size());

    for (const Point& p : allPeaks)
    {
        bool keep = true;
        for (const Point& q : filtered)
        {
            int dx = p.x - q.x;
            int dy = p.y - q.y;
            if (dx * dx + dy * dy < minDist2)
            {
                keep = false;
                break;
            }
        }
        if (keep) filtered.push_back(p);
    }

    return filtered;
}

Point findNearestSkeletonPointFast(const Point& pt, const vector<Point>& skeletonPts)
{
    int minDist2 = INT_MAX;
    Point best = pt;

    for (const Point& p : skeletonPts)
    {
        int dx = p.x - pt.x;
        int dy = p.y - pt.y;
        int d2 = dx * dx + dy * dy;
        if (d2 < minDist2)
        {
            minDist2 = d2;
            best = p;
        }
    }
    return best;
}

Mat bfsDistanceMapOnSkeleton(const Mat& skeleton, const Point& start)
{
    Mat dist(skeleton.size(), CV_32S, Scalar(-1));
    if (!insideImage(start.x, start.y, skeleton.cols, skeleton.rows)) return dist;
    if (skeleton.at<uchar>(start.y, start.x) == 0) return dist;

    vector<Point> q;
    q.reserve((size_t)skeleton.rows * skeleton.cols / 4);
    q.push_back(start);
    dist.at<int>(start.y, start.x) = 0;

    size_t head = 0;
    while (head < q.size())
    {
        Point u = q[head++];
        int du = dist.at<int>(u.y, u.x);

        for (int k = 0; k < 8; ++k)
        {
            int nx = u.x + dx8[k];
            int ny = u.y + dy8[k];
            if (!insideImage(nx, ny, skeleton.cols, skeleton.rows)) continue;
            if (skeleton.at<uchar>(ny, nx) == 0) continue;
            if (dist.at<int>(ny, nx) != -1) continue;

            dist.at<int>(ny, nx) = du + 1;
            q.push_back(Point(nx, ny));
        }
    }

    return dist;
}

bool getShortestPathOnSkeleton(const Mat& skeleton, const Point& start, const Point& end, vector<Point>& path)
{
    path.clear();
    if (start == end)
    {
        path.push_back(start);
        return true;
    }
    if (!insideImage(start.x, start.y, skeleton.cols, skeleton.rows) ||
        !insideImage(end.x, end.y, skeleton.cols, skeleton.rows))
        return false;
    if (skeleton.at<uchar>(start.y, start.x) == 0 || skeleton.at<uchar>(end.y, end.x) == 0)
        return false;

    int rows = skeleton.rows;
    int cols = skeleton.cols;
    auto idOf = [cols](const Point& p) { return p.y * cols + p.x; };

    vector<uchar> visited((size_t)rows * cols, 0);
    vector<Point> parent((size_t)rows * cols, Point(-1, -1));
    vector<Point> q;
    q.reserve((size_t)rows * cols / 4);

    visited[idOf(start)] = 1;
    parent[idOf(start)] = Point(-2, -2);
    q.push_back(start);

    bool found = false;
    size_t head = 0;
    while (head < q.size() && !found)
    {
        Point u = q[head++];
        for (int k = 0; k < 8; ++k)
        {
            int nx = u.x + dx8[k];
            int ny = u.y + dy8[k];
            if (!insideImage(nx, ny, cols, rows)) continue;
            if (skeleton.at<uchar>(ny, nx) == 0) continue;

            Point v(nx, ny);
            int vid = idOf(v);
            if (visited[vid]) continue;

            visited[vid] = 1;
            parent[vid] = u;

            if (v == end)
            {
                found = true;
                break;
            }
            q.push_back(v);
        }
    }

    if (!found) return false;

    Point cur = end;
    while (cur != Point(-2, -2))
    {
        path.push_back(cur);
        cur = parent[idOf(cur)];
    }
    reverse(path.begin(), path.end());
    return true;
}

struct MstEdge
{
    int u = -1;
    int v = -1;
    int weight = 0;
};

vector<MstEdge> getMSTEdgesFast(const Mat& skeleton, const vector<Point>& skeletonTargets)
{
    int n = (int)skeletonTargets.size();
    if (n <= 1) return {};

    struct Edge
    {
        int u, v, weight;
        bool operator<(const Edge& other) const
        {
            return weight < other.weight;
        }
    };

    vector<Edge> edges;

    for (int i = 0; i < n; ++i)
    {
        Mat dist = bfsDistanceMapOnSkeleton(skeleton, skeletonTargets[i]);
        for (int j = i + 1; j < n; ++j)
        {
            int d = dist.at<int>(skeletonTargets[j].y, skeletonTargets[j].x);
            if (d > 0) edges.push_back({ i, j, d });
        }
    }

    stable_sort(edges.begin(), edges.end());

    vector<int> parent(n);
    for (int i = 0; i < n; ++i) parent[i] = i;

    function<int(int)> findRoot = [&](int x)
        {
            return parent[x] == x ? x : parent[x] = findRoot(parent[x]);
        };

    vector<MstEdge> mst;
    mst.reserve(max(0, n - 1));

    for (const Edge& e : edges)
    {
        int ru = findRoot(e.u);
        int rv = findRoot(e.v);
        if (ru == rv) continue;

        parent[ru] = rv;
        mst.push_back({ e.u, e.v, e.weight });
        if ((int)mst.size() == n - 1) break;
    }

    return mst;
}

Point2f contourCenterPoint(const vector<Point>& contour)
{
    if (contour.empty()) return Point2f(0.0f, 0.0f);

    Moments m = moments(contour);
    if (fabs(m.m00) > 1e-6)
        return Point2f((float)(m.m10 / m.m00), (float)(m.m01 / m.m00));

    Rect box = boundingRect(contour);
    return Point2f(box.x + box.width * 0.5f, box.y + box.height * 0.5f);
}

vector<MstEdge> getClosedRingEdgesByAngle(
    const vector<Point>& mappedCentersLocal,
    const vector<Point>& largestHoleContour)
{
    int n = (int)mappedCentersLocal.size();
    vector<MstEdge> ringEdges;
    if (n <= 1) return ringEdges;

    Point2f holeCenter = contourCenterPoint(largestHoleContour);

    vector<int> order(n);
    for (int i = 0; i < n; ++i) order[i] = i;

    stable_sort(order.begin(), order.end(),
        [&](int a, int b)
        {
            double aa = atan2((double)mappedCentersLocal[a].y - holeCenter.y,
                (double)mappedCentersLocal[a].x - holeCenter.x);
            double bb = atan2((double)mappedCentersLocal[b].y - holeCenter.y,
                (double)mappedCentersLocal[b].x - holeCenter.x);
            return aa < bb;
        });

    if (n == 2)
    {
        
        ringEdges.push_back({ order[0], order[1], 0 });
        return ringEdges;
    }

    ringEdges.reserve(n);
    for (int k = 0; k < n; ++k)
    {
        int u = order[k];
        int v = order[(k + 1) % n];
        ringEdges.push_back({ u, v, 0 });
    }

    return ringEdges;
}

bool getAlternativePathOnRingSkeleton(
    const Mat& skeleton,
    const Point& start,
    const Point& end,
    const vector<Point>& firstPath,
    vector<Point>& alternativePath)
{
    alternativePath.clear();
    if (firstPath.size() < 5) return false;

    Mat skeletonWithoutFirstPath = skeleton.clone();

    const int keepEndLen = 3;
    for (int i = keepEndLen; i < (int)firstPath.size() - keepEndLen; ++i)
    {
        const Point& p = firstPath[i];
        if (insideImage(p.x, p.y, skeletonWithoutFirstPath.cols, skeletonWithoutFirstPath.rows))
            skeletonWithoutFirstPath.at<uchar>(p.y, p.x) = 0;
    }

    return getShortestPathOnSkeleton(skeletonWithoutFirstPath, start, end, alternativePath);
}

int contourIndexDistance(int idx1, int idx2, int contourSize)
{
    int d = abs(idx1 - idx2);
    return min(d, contourSize - d);
}

float point2SegmentDist(Point p, Point a, Point b)
{
    Point ap = p - a;
    Point ab = b - a;
    float denom = (float)(ab.x * ab.x + ab.y * ab.y);
    if (denom < 1e-6f) return (float)norm(p - a);

    float t = (ap.x * ab.x + ap.y * ab.y) / denom;
    t = max(0.0f, min(1.0f, t));

    Point proj(cvRound(a.x + t * ab.x), cvRound(a.y + t * ab.y));
    return (float)norm(p - proj);
}

bool isPointInCenterOrthogonalRoiSearchArea(
    const Point& p,
    const Point& centerA,
    const Point& centerB)
{
    Point2f A((float)centerA.x, (float)centerA.y);
    Point2f B((float)centerB.x, (float)centerB.y);
    Point2f P((float)p.x, (float)p.y);

    Point2f v = B - A;
    double len2 = (double)v.x * v.x + (double)v.y * v.y;
    if (len2 < 1e-6) return false;

    double t = ((P.x - A.x) * v.x + (P.y - A.y) * v.y) / len2;

    const double eps = 1e-6;
    return t >= -eps && t <= 1.0 + eps;
}

float crossLine(Point2f A, Point2f B, Point2f P)
{
    return (float)((B.x - A.x) * (P.y - A.y) - (B.y - A.y) * (P.x - A.x));
}

double cross2D(Point2f a, Point2f b, Point2f c)
{
    return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
}

bool pointInSegmentBox(Point2f p, Point2f a, Point2f b)
{
    const double eps = 1e-6;
    return p.x >= min(a.x, b.x) - eps &&
        p.x <= max(a.x, b.x) + eps &&
        p.y >= min(a.y, b.y) - eps &&
        p.y <= max(a.y, b.y) + eps;
}

bool segmentIntersectionPoint(Point2f p1, Point2f p2, Point2f q1, Point2f q2, Point2f& inter)
{
    Point2f r = p2 - p1;
    Point2f s = q2 - q1;
    double denom = r.x * s.y - r.y * s.x;

    if (fabs(denom) < 1e-6)
    {
        if (fabs(cross2D(p1, p2, q1)) < 1e-6)
        {
            if (pointInSegmentBox(q1, p1, p2)) { inter = q1; return true; }
            if (pointInSegmentBox(q2, p1, p2)) { inter = q2; return true; }
            if (pointInSegmentBox(p1, q1, q2)) { inter = p1; return true; }
            if (pointInSegmentBox(p2, q1, q2)) { inter = p2; return true; }
        }
        return false;
    }

    Point2f qp = q1 - p1;
    double t = (qp.x * s.y - qp.y * s.x) / denom;
    double u = (qp.x * r.y - qp.y * r.x) / denom;

    if (t >= -1e-6 && t <= 1.0 + 1e-6 &&
        u >= -1e-6 && u <= 1.0 + 1e-6)
    {
        inter = p1 + (float)t * r;
        return true;
    }

    return false;
}

bool findDentLinePathIntersectionByGeometry(
    const vector<Point>& path,
    Point p1,
    Point p2,
    int& bestPathIdx,
    Point2f& bestInterPoint)
{
    bestPathIdx = -1;
    bestInterPoint = Point(-1, -1);
    if (path.size() < 2) return false;

    Point2f fp1((float)p1.x, (float)p1.y);
    Point2f fp2((float)p2.x, (float)p2.y);
    float totalDist = (float)norm(p2 - p1);
    float bestTDist = 1e9f;

    for (int i = 0; i < (int)path.size() - 1; ++i)
    {
        Point2f fq1((float)path[i].x, (float)path[i].y);
        Point2f fq2((float)path[i + 1].x, (float)path[i + 1].y);

        Point2f inter;
        if (!segmentIntersectionPoint(fp1, fp2, fq1, fq2, inter)) continue;

        Point interRound(cvRound(inter.x), cvRound(inter.y));
        float t = 0.0f;
        if (totalDist > 1e-6f)
        {
            t = (float)norm(interRound - p1) / totalDist;
            t = max(0.0f, min(1.0f, t));
        }

        float d = fabs(t - 0.5f);
        if (d < bestTDist)
        {
            bestTDist = d;
            bestPathIdx = i;
            bestInterPoint = interRound;
        }
    }

    return bestPathIdx >= 0;
}

bool getLocalPathDirectionLine(const vector<Point>& path, int idx, Point& Q1, Point& Q2, int halfLen = 1)
{
    if (path.size() < 2) return false;
    if (idx < 0 || idx >= (int)path.size()) return false;

    int i1 = max(0, idx - halfLen);
    int i2 = min((int)path.size() - 1, idx + halfLen);
    if (i1 == i2) return false;

    Q1 = path[i1];
    Q2 = path[i2];
    return true;
}

struct DentInfo
{
    Point pt;
    int contourIdx;
};

struct DentPairCandidate
{
    int idx1 = -1;
    int idx2 = -1;

    Point d1, d2;             
    Point origD1, origD2;     

    float score = 1e9f;
    bool valid = false;
};

struct FeatureDetectionResult
{
    vector<DentInfo> all;         
    vector<DentInfo> simplified;  
};

FeatureDetectionResult detectContourFeaturePointsDetailed(const vector<Point>& contour, bool useNegativeCross)
{
    FeatureDetectionResult out;
    vector<int> featureIdx;

    int n = (int)contour.size();
    if (n < 2 * CONCAVE_STEP + 1) return out;

    for (int j = 0; j < n; ++j)
    {
        int prevIdx = (j - CONCAVE_STEP + n) % n;
        int nextIdx = (j + CONCAVE_STEP) % n;

        Point qian = contour[prevIdx];
        Point zhong = contour[j];
        Point hou = contour[nextIdx];

        Point v1 = qian - zhong;
        Point v2 = hou - zhong;

        double dot = v1.x * v2.x + v1.y * v2.y;
        double mod1 = sqrt((double)v1.x * v1.x + (double)v1.y * v1.y);
        double mod2 = sqrt((double)v2.x * v2.x + (double)v2.y * v2.y);

        double cosAngle = dot / (mod1 * mod2 + 1e-6);
        cosAngle = max(-1.0, min(1.0, cosAngle));
        double angle = acos(cosAngle) * 180.0 / CV_PI;

        double cr = v1.x * v2.y - v2.x * v1.y;
        bool crossOK = useNegativeCross ? (cr < 0.0) : (cr > 0.0);

        if (angle < CONCAVE_ANGLE_THRESH && crossOK)
        {
            featureIdx.push_back(j);
            out.all.push_back({ contour[j], j });
        }
    }

    if (featureIdx.empty()) return out;

    vector<vector<int>> groups;
    vector<int> group;
    group.push_back(featureIdx[0]);
    for (size_t i = 1; i < featureIdx.size(); ++i)
    {
        if (featureIdx[i] - group.back() < CONCAVE_GROUP_GAP)
            group.push_back(featureIdx[i]);
        else
        {
            groups.push_back(std::move(group));
            group = { featureIdx[i] };
        }
    }
    groups.push_back(std::move(group));

    if (groups.size() > 1 &&
        featureIdx.front() + n - featureIdx.back() < CONCAVE_GROUP_GAP)
    {

        vector<int>& lastGroup = groups.back();
        const vector<int>& firstGroup = groups.front();
        lastGroup.insert(lastGroup.end(), firstGroup.begin(), firstGroup.end());
        groups.erase(groups.begin());
    }

    for (const vector<int>& indices : groups)
    {
        const int mid = indices[indices.size() / 2];
        out.simplified.push_back({ contour[mid], mid });
    }

    sort(out.simplified.begin(), out.simplified.end(),
        [](const DentInfo& a, const DentInfo& b)
        {
            return a.contourIdx < b.contourIdx;
        });

    return out;
}

vector<DentInfo> detectContourFeaturePoints(const vector<Point>& contour, bool useNegativeCross)
{
    return detectContourFeaturePointsDetailed(contour, useNegativeCross).simplified;
}

vector<DentInfo> detectConcavePoints(const vector<Point>& contour)
{
    return detectContourFeaturePoints(contour, true);
}

vector<DentInfo> detectInnerConvexPoints(const vector<Point>& contour)
{
    return detectContourFeaturePoints(contour, true);
}

bool getOuterAndLargestHoleContours(
    const Mat& roiMask,
    vector<Point>& outerContour,
    vector<Point>& largestHoleContour)
{
    outerContour.clear();
    largestHoleContour.clear();

    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(roiMask.clone(), contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_NONE);

    if (contours.empty() || hierarchy.empty()) return false;

    int outerIdx = -1;
    double maxOuterArea = -1.0;

    for (int i = 0; i < (int)contours.size(); ++i)
    {
        if (hierarchy[i][3] != -1) continue; 

        double area = fabs(contourArea(contours[i]));
        if (area > maxOuterArea)
        {
            maxOuterArea = area;
            outerIdx = i;
        }
    }

    if (outerIdx < 0) return false;
    outerContour = contours[outerIdx];

    int bestHoleIdx = -1;
    double maxHoleArea = -1.0;

    for (int child = hierarchy[outerIdx][2]; child != -1; child = hierarchy[child][0])
    {
        double area = fabs(contourArea(contours[child]));
        if (area > maxHoleArea)
        {
            maxHoleArea = area;
            bestHoleIdx = child;
        }
    }

    const double minHoleArea = 200.0;
    if (bestHoleIdx >= 0 && maxHoleArea >= minHoleArea)
    {
        largestHoleContour = contours[bestHoleIdx];
        return true;
    }

    return false;
}

bool evaluateDentPairOnPath(
    const vector<Point>& path,
    Point p1,
    Point p2,
    int idx1,
    int idx2,
    int contourIdx1,
    int contourIdx2,
    bool sameContourPair,
    int contourSize,
    const Rect& roiRect,
    DentPairCandidate& cand)
{
    int interIdx = -1;
    Point2f interPoint;
    const bool hasIntersection =
        findDentLinePathIntersectionByGeometry(path, p1, p2, interIdx, interPoint);
    if (!hasIntersection) return false;

    Point Q1, Q2;
    if (!getLocalPathDirectionLine(path, interIdx, Q1, Q2))
        return false;

    float dist1 = point2SegmentDist(p1, Q1, Q2);
    float dist2 = point2SegmentDist(p2, Q1, Q2);

    bool absorbed1 = (dist1 < 0.001f);
    bool absorbed2 = (dist2 < 0.001f);

    Point pt1 = p1;
    Point pt2 = p2;

    if (absorbed1)
    {
        Point ap = p1 - Q1;
        Point ab = Q2 - Q1;
        float denom = (float)(ab.x * ab.x + ab.y * ab.y);
        float t = (ap.x * ab.x + ap.y * ab.y) / (denom + 1e-6f);
        t = max(0.0f, min(1.0f, t));
        pt1 = Point(cvRound(Q1.x + t * ab.x), cvRound(Q1.y + t * ab.y));
    }

    if (absorbed2)
    {
        Point ap = p2 - Q1;
        Point ab = Q2 - Q1;
        float denom = (float)(ab.x * ab.x + ab.y * ab.y);
        float t = (ap.x * ab.x + ap.y * ab.y) / (denom + 1e-6f);
        t = max(0.0f, min(1.0f, t));
        pt2 = Point(cvRound(Q1.x + t * ab.x), cvRound(Q1.y + t * ab.y));
    }

    float cr1 = crossLine(Q1, Q2, p1);
    float cr2 = crossLine(Q1, Q2, p2);
    if (cr1 * cr2 > 1.0f) return false;

    if (sameContourPair && absorbed1 && absorbed2)
    {
        double eucDist = norm(p1 - p2);
        if (eucDist >= 30.0) return false;

        int cDist = contourIndexDistance(contourIdx1, contourIdx2, contourSize);
        if (cDist <= 100) return false;
    }

    float score = dist1 + dist2;

    cand.valid = true;
    cand.idx1 = idx1;
    cand.idx2 = idx2;
    cand.d1 = Point(pt1.x + roiRect.x, pt1.y + roiRect.y);
    cand.d2 = Point(pt2.x + roiRect.x, pt2.y + roiRect.y);
    cand.origD1 = Point(p1.x + roiRect.x, p1.y + roiRect.y);
    cand.origD2 = Point(p2.x + roiRect.x, p2.y + roiRect.y);
    cand.score = score;

    return true;
}

DentPairCandidate findBestDentPairForPath(
    const vector<Point>& path,
    const vector<DentInfo>& dentsLocalInfo,
    int contourSize,
    const Rect& roiRect,
    const Point& centerA,
    const Point& centerB)
{
    DentPairCandidate best;
    if (path.empty() || dentsLocalInfo.size() < 2) return best;

    vector<int> nearbyDentIdx;
    nearbyDentIdx.reserve(dentsLocalInfo.size());
    for (int i = 0; i < (int)dentsLocalInfo.size(); ++i)
    {
        if (isPointInCenterOrthogonalRoiSearchArea(
            dentsLocalInfo[i].pt, centerA, centerB))
            nearbyDentIdx.push_back(i);
    }

    if (nearbyDentIdx.size() < 2) return best;

    for (int a = 0; a < (int)nearbyDentIdx.size(); ++a)
    {
        for (int b = a + 1; b < (int)nearbyDentIdx.size(); ++b)
        {
            int i1 = nearbyDentIdx[a];
            int i2 = nearbyDentIdx[b];

            DentPairCandidate cur;
            if (!evaluateDentPairOnPath(
                path,
                dentsLocalInfo[i1].pt,
                dentsLocalInfo[i2].pt,
                i1,
                i2,
                dentsLocalInfo[i1].contourIdx,
                dentsLocalInfo[i2].contourIdx,
                true,
                contourSize,
                roiRect,
                cur))
                continue;

            if (!best.valid || cur.score < best.score)
                best = cur;
        }
    }

    return best;
}

DentPairCandidate findBestOuterInnerPairForPath(
    const vector<Point>& path,
    const vector<DentInfo>& outerConcaveInfo,
    const vector<DentInfo>& innerConvexInfo,
    const Rect& roiRect,
    const Point& centerA,
    const Point& centerB)
{
    DentPairCandidate best;
    if (path.empty() || outerConcaveInfo.empty() || innerConvexInfo.empty()) return best;

    vector<int> nearbyOuterIdx;
    vector<int> nearbyInnerIdx;
    nearbyOuterIdx.reserve(outerConcaveInfo.size());
    nearbyInnerIdx.reserve(innerConvexInfo.size());

    for (int i = 0; i < (int)outerConcaveInfo.size(); ++i)
    {
        if (isPointInCenterOrthogonalRoiSearchArea(
            outerConcaveInfo[i].pt, centerA, centerB))
            nearbyOuterIdx.push_back(i);
    }

    for (int i = 0; i < (int)innerConvexInfo.size(); ++i)
    {
        if (isPointInCenterOrthogonalRoiSearchArea(
            innerConvexInfo[i].pt, centerA, centerB))
            nearbyInnerIdx.push_back(i);
    }

    if (nearbyOuterIdx.empty() || nearbyInnerIdx.empty()) return best;

    for (int oi : nearbyOuterIdx)
    {
        for (int ii : nearbyInnerIdx)
        {
            DentPairCandidate cur;
            if (!evaluateDentPairOnPath(
                path,
                outerConcaveInfo[oi].pt,
                innerConvexInfo[ii].pt,
                oi,
                ii,
                outerConcaveInfo[oi].contourIdx,
                innerConvexInfo[ii].contourIdx,
                false,
                0,
                roiRect,
                cur))
                continue;

            if (!best.valid || cur.score < best.score)
                best = cur;
        }
    }

    return best;
}

void drawSplitParticleBoundaries(
    const Mat& roiMask,
    const Rect& roiRect,
    const vector<pair<Point, Point>>& selectedSplitLinesOriginalGlobal,
    vector<vector<Point>>& predContoursGlobal)
{
    if (roiMask.empty()) return;

    Mat splitMask = roiMask.clone();
    for (const auto& seg : selectedSplitLinesOriginalGlobal)
    {
        Point p1(seg.first.x - roiRect.x, seg.first.y - roiRect.y);
        Point p2(seg.second.x - roiRect.x, seg.second.y - roiRect.y);
        line(splitMask, p1, p2, Scalar(0), 2, LINE_8);
    }

    Mat subLabels, subStats, subCentroids;
    int subCount = connectedComponentsWithStats(splitMask, subLabels, subStats, subCentroids, 8);

    for (int label = 1; label < subCount; ++label)
    {
        int area = subStats.at<int>(label, CC_STAT_AREA);
        if (area <= 0) continue;

        Mat oneMask = (subLabels == label);
        oneMask.convertTo(oneMask, CV_8U, 255);

        vector<vector<Point>> subContours;
        findContours(oneMask, subContours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        for (auto contour : subContours)
        {
            for (Point& p : contour)
            {
                p.x += roiRect.x;
                p.y += roiRect.y;
            }
            predContoursGlobal.push_back(contour);
        }
    }
}

void processConnectedComponent(
    const Mat& roiMask,
    const Rect& roiRect,
    vector<vector<Point>>& predContoursGlobal,
    vector<pair<Point, Point>>& finalSegLinesGlobal,
    CenterPeakStats& peakStats)
{
    vector<Point> outerContour;
    vector<Point> largestHoleContour;
    bool hasLargestHole = getOuterAndLargestHoleContours(roiMask, outerContour, largestHoleContour);

    if (outerContour.empty()) return;

    FeatureDetectionResult outerFeature = detectContourFeaturePointsDetailed(outerContour, true);
    vector<DentInfo> outerConcaveInfo = outerFeature.simplified;

    FeatureDetectionResult innerFeature;
    vector<DentInfo> innerConvexInfo;
    bool ringMode = false;
    if (hasLargestHole && !largestHoleContour.empty())
    {
        innerFeature = detectContourFeaturePointsDetailed(largestHoleContour, true);
        innerConvexInfo = innerFeature.simplified;
        ringMode = (!outerConcaveInfo.empty() && !innerConvexInfo.empty());
    }

    if (!ringMode && outerConcaveInfo.size() < 2)
    {
        drawSplitParticleBoundaries(roiMask, roiRect, {}, predContoursGlobal);
        return;
    }

    if (hasLargestHole && !ringMode)
    {
        drawSplitParticleBoundaries(roiMask, roiRect, {}, predContoursGlobal);
        return;
    }

    Mat dist;
    distanceTransform(roiMask, dist, DIST_L2, 3);

    int rawPeaksInComponent = 0;
    vector<Point> originalCentersLocal = detectLocalMaximaPerBlobFast(dist, &rawPeaksInComponent);
    peakStats.rawPeakCount += rawPeaksInComponent;
    peakStats.usedCenterCount += (long long)originalCentersLocal.size();
    if (originalCentersLocal.size() < 2)
    {
        drawSplitParticleBoundaries(roiMask, roiRect, {}, predContoursGlobal);
        return;
    }

    Mat skeleton;
    threshold(dist, skeleton, 0.001, 255, THRESH_BINARY);
    skeleton.convertTo(skeleton, CV_8U);
    thinningZhangSuen(skeleton);

    vector<Point> skeletonPts;
    findNonZero(skeleton, skeletonPts);
    if (skeletonPts.empty())
    {
        drawSplitParticleBoundaries(roiMask, roiRect, {}, predContoursGlobal);
        return;
    }

    vector<Point> skeletonTargets;
    skeletonTargets.reserve(originalCentersLocal.size());
    for (const Point& c : originalCentersLocal)
        skeletonTargets.push_back(findNearestSkeletonPointFast(c, skeletonPts));

    vector<MstEdge> graphEdges;
    if (ringMode)
    {
        graphEdges = getClosedRingEdgesByAngle(skeletonTargets, largestHoleContour);
    }
    else
    {
        if (skeletonTargets.size() == 2)
            graphEdges.push_back({ 0, 1, 0 });
        else
            graphEdges = getMSTEdgesFast(skeleton, skeletonTargets);
    }

    if (graphEdges.empty())
    {
        drawSplitParticleBoundaries(roiMask, roiRect, {}, predContoursGlobal);
        return;
    }

    vector<DentPairCandidate> candidates;
    candidates.reserve(graphEdges.size() + 1);
    for (const MstEdge& e : graphEdges)
    {
        Point centerA = skeletonTargets[e.u];
        Point centerB = skeletonTargets[e.v];

        vector<Point> path;
        if (!getShortestPathOnSkeleton(skeleton, skeletonTargets[e.u], skeletonTargets[e.v], path))
            continue;

        vector<vector<Point>> pathsToEvaluate;
        pathsToEvaluate.push_back(path);

        if (ringMode && skeletonTargets.size() == 2)
        {
            vector<Point> alternativePath;
            if (getAlternativePathOnRingSkeleton(
                skeleton,
                skeletonTargets[e.u],
                skeletonTargets[e.v],
                path,
                alternativePath))
            {
                pathsToEvaluate.push_back(alternativePath);
            }
        }

        for (const vector<Point>& curPath : pathsToEvaluate)
        {
            DentPairCandidate cand;
            if (ringMode)
            {
                cand = findBestOuterInnerPairForPath(
                    curPath,
                    outerConcaveInfo,
                    innerConvexInfo,
                    roiRect,
                    centerA,
                    centerB);
            }
            else
            {
                cand = findBestDentPairForPath(
                    curPath,
                    outerConcaveInfo,
                    (int)outerContour.size(),
                    roiRect,
                    centerA,
                    centerB);
            }

            if (cand.valid) candidates.push_back(cand);
        }
    }

    stable_sort(candidates.begin(), candidates.end(),
        [](const DentPairCandidate& a, const DentPairCandidate& b)
        {
            return a.score < b.score;
        });

    set<int> usedOuterIdx;
    set<int> usedInnerIdx;
    set<int> usedDentIdx;
    vector<pair<Point, Point>> selectedSplitLinesOriginalGlobal;

    for (const DentPairCandidate& cand : candidates)
    {
        if (ringMode)
        {
            if (usedOuterIdx.count(cand.idx1) || usedInnerIdx.count(cand.idx2))
                continue;

            usedOuterIdx.insert(cand.idx1);
            usedInnerIdx.insert(cand.idx2);
        }
        else
        {
            if (usedDentIdx.count(cand.idx1) || usedDentIdx.count(cand.idx2))
                continue;

            usedDentIdx.insert(cand.idx1);
            usedDentIdx.insert(cand.idx2);
        }

        selectedSplitLinesOriginalGlobal.push_back({ cand.origD1, cand.origD2 });
        finalSegLinesGlobal.push_back({ cand.origD1, cand.origD2 });
    }

    drawSplitParticleBoundaries(roiMask, roiRect, selectedSplitLinesOriginalGlobal, predContoursGlobal);
}

Mat drawFinalSegLinesOnOriginal(
    const Mat& src,
    const vector<pair<Point, Point>>& finalSegLinesGlobal)
{
    Mat finalImg = src.clone();
    if (finalImg.channels() == 1)
        cvtColor(finalImg, finalImg, COLOR_GRAY2BGR);

    const int finalSegLineWidth = 5;
    for (const auto& seg : finalSegLinesGlobal)
    {
        line(finalImg, seg.first, seg.second,
            Scalar(255, 255, 255), finalSegLineWidth, LINE_AA);
    }
    return finalImg;
}

json contourToLabelmePointsForPred(const vector<Point>& contour)
{
    json pts = json::array();
    for (const Point& p : contour)
        pts.push_back({ (double)p.x, (double)p.y });
    return pts;
}

bool savePredContoursToLabelmeJson(
    const string& imagePath,
    const string& jsonSavePath,
    const vector<vector<Point>>& predContours,
    int imageWidth,
    int imageHeight)
{
    json j;
    j["version"] = "5.7.0";
    j["flags"] = json::object();
    j["shapes"] = json::array();

    for (const auto& contour : predContours)
    {
        if (contour.size() < 3) continue;

        json shape;
        shape["label"] = "particle";
        shape["points"] = contourToLabelmePointsForPred(contour);
        shape["group_id"] = nullptr;
        shape["description"] = "";
        shape["shape_type"] = "polygon";
        shape["flags"] = json::object();
        shape["mask"] = nullptr;
        j["shapes"].push_back(shape);
    }

    j["imagePath"] = fs::path(imagePath).filename().string();
    j["imageData"] = nullptr;
    j["imageHeight"] = imageHeight;
    j["imageWidth"] = imageWidth;

    fs::create_directories(fs::path(jsonSavePath).parent_path());
    ofstream ofs(jsonSavePath);
    if (!ofs.is_open())
    {
        cout << "Failed to open predicted JSON for writing: " << jsonSavePath << endl;
        return false;
    }
    ofs << j.dump(2);
    ofs.close();
    if (!ofs.good()) {
        cout << "Failed to write predicted JSON: " << jsonSavePath << endl;
        return false;
    }
    cout << "Predicted JSON saved: " << jsonSavePath << endl;
    return true;
}

static const double FINAL_IOU_THRESH = 0.95;
static const double FERET_LONG_REL_ERR_THRESH = 0.05;
static const double MATCH_IOU_THRESH = 0.30;
static const double UNDER_GT_COVER_THRESH = 0.30;
static const double OVER_PRED_INSIDE_THRESH = 0.30;
static const int MIN_EVAL_AREA = 20;

struct EvalParticle {
    int id = -1;
    vector<Point> contour;
    Mat mask;
    int areaPx = 0;
    double feretLongDiameter = 0.0;
};
struct EvalPairCandidate {
    int gtIdx = -1, predIdx = -1;
    double interArea = 0.0, iou = 0.0;
};
struct EvalMatchResult {
    int gtIdx = -1, predIdx = -1;
    double interArea = 0.0, gtArea = 0.0, predArea = 0.0;
    double iou = 0.0, dice = 0.0, precision = 0.0, recall = 0.0;
    bool iouOK = false;
    double gtLongDiameter = 0.0, predLongDiameter = 0.0;
    double longRelErr = 1e9;
    bool longDiameterOK = false, finalOK = false;
};
struct EvalImageResult {
    string imageId;
    int gtCount = 0, predCount = 0, matchedCount = 0;
    int iouOKCount = 0, iouBadCount = 0, longOKCount = 0, longBadCount = 0;
    int finalOKCount = 0, underSegPredCount = 0, overSegGtCount = 0;
    int missedGtCount = 0, falsePredCount = 0;
    int evaluationItemCount = 0; 
    int instanceFalsePositiveCount = 0, instanceFalseNegativeCount = 0;
    
    double sumIou = 0.0, sumDice = 0.0, sumPrecision = 0.0, sumRecall = 0.0, sumLongRelErr = 0.0;
    double avgIou = 0.0, avgDice = 0.0, avgPrecision = 0.0, avgRecall = 0.0;
    double avgLongRelErr = 0.0, accuracy = 0.0;
    double instancePrecision = 0.0, instanceRecall = 0.0, instanceF1 = 0.0;
};

vector<fs::path> getEvalImageFiles(const fs::path& dir)
{
    vector<fs::path> images;
    if (!fs::exists(dir) || !fs::is_directory(dir)) return images;

    const int startImgId = 1;
    const int endImgId = 201;
    for (int id = startImgId; id < endImgId; ++id) {
        const fs::path file = dir / (to_string(id) + ".jpg");
        if (fs::exists(file) && fs::is_regular_file(file)) {
            images.push_back(file);
        }
        else {
            cerr << "Warning: input image is missing: " << file.string() << " (skipping this ID).\n";
        }
    }
    return images;
}

bool evalAcceptedLabel(const string& label)
{
    return label == "particle" || label == "Particle" || label == "颗粒";
}

double evalFeretLong(const vector<Point>& contour)
{
    if (contour.size() < 2) return 0;
    vector<Point> hull;
    convexHull(contour, hull);
    double maxD2 = 0;
    for (size_t i = 0; i < hull.size(); ++i)
        for (size_t j = i + 1; j < hull.size(); ++j) {
            double dx = double(hull[i].x) - hull[j].x;
            double dy = double(hull[i].y) - hull[j].y;
            maxD2 = max(maxD2, dx * dx + dy * dy);
        }
    return sqrt(maxD2);
}

bool loadEvalParticles(const fs::path& jsonPath, Size imageSize,
    vector<EvalParticle>& particles, string& reason)
{
    particles.clear(); reason.clear();
    ifstream in(jsonPath, ios::binary);
    if (!in) { reason = "JSON file is missing or cannot be opened"; return false; }
    try {
        json j; in >> j;
        if (!j.is_object()) { reason = "The JSON root must be an object"; return false; }
        if (j.contains("imageWidth") && j.contains("imageHeight")) {
            const int w = j.at("imageWidth").get<int>();
            const int h = j.at("imageHeight").get<int>();
            if (w != imageSize.width || h != imageSize.height) {
                reason = "JSON image dimensions do not match the input image";
                return false;
            }
        }
        if (!j.contains("shapes") || !j.at("shapes").is_array()) {
            reason = "JSON does not contain a shapes array"; return false;
        }
        vector<vector<Point>> polygons;
        auto readShapes = [&](bool fallback) {
            for (const auto& shape : j.at("shapes")) {
                if (!shape.is_object()) continue;
                string label = shape.value("label", string(""));
                if (shape.value("shape_type", string("")) != "polygon") continue;
                if (fallback) {
                    if (label == "cut_line" || label == "cluster_inner" ||
                        label == "cluster_outer" || label == "cluster") continue;
                }
                else if (!evalAcceptedLabel(label)) continue;
                if (!shape.contains("points") || !shape.at("points").is_array()) continue;
                vector<Point> contour;
                for (const auto& pt : shape.at("points")) {
                    if (!pt.is_array() || pt.size() < 2) continue;
                    contour.emplace_back(cvRound(pt[0].get<double>()), cvRound(pt[1].get<double>()));
                }
                if (contour.size() >= 3) polygons.push_back(std::move(contour));
            }
            };
        readShapes(false);
        if (polygons.empty()) readShapes(true);
        int nextID = 0;
        for (const auto& contour : polygons) {
            Mat mask = Mat::zeros(imageSize, CV_8U);
            fillPoly(mask, vector<vector<Point>>{contour}, Scalar(255));
            const int area = countNonZero(mask);
            if (area < MIN_EVAL_AREA) continue;
            EvalParticle p;
            p.id = nextID++; p.contour = contour; p.mask = std::move(mask);
            p.areaPx = area; p.feretLongDiameter = evalFeretLong(contour);
            particles.push_back(std::move(p));
        }
        return true;
    }
    catch (const exception& ex) {
        reason = string("Invalid JSON or annotation data: ") + ex.what();
        particles.clear(); return false;
    }
}

double evalIntersectionArea(const Mat& a, const Mat& b)
{
    Mat inter;
    bitwise_and(a, b, inter);
    return double(countNonZero(inter));
}

void evalFinalizeMetrics(EvalImageResult& r)
{
    r.evaluationItemCount = r.matchedCount + r.missedGtCount + r.falsePredCount;
    r.avgIou = r.avgLongRelErr = 0.0;
    if (r.evaluationItemCount > 0) {
        const double n = double(r.evaluationItemCount);
        r.avgIou = r.sumIou / n;
        r.avgLongRelErr = (r.sumLongRelErr + r.missedGtCount + r.falsePredCount) / n;
    }
    
    r.avgDice = r.avgPrecision = r.avgRecall = 0.0;
    if (r.matchedCount > 0) {
        const double m = double(r.matchedCount);
        r.avgDice = r.sumDice / m;
        r.avgPrecision = r.sumPrecision / m;
        r.avgRecall = r.sumRecall / m;
    }
    r.accuracy = r.gtCount > 0 ? double(r.finalOKCount) / r.gtCount : 0.0;
    r.instancePrecision = r.predCount > 0 ? double(r.finalOKCount) / r.predCount : 0.0;
    r.instanceRecall = r.accuracy;
    r.instanceFalsePositiveCount = r.predCount - r.finalOKCount;
    r.instanceFalseNegativeCount = r.gtCount - r.finalOKCount;
    const double instanceDenom = double(r.gtCount) + r.predCount;
    r.instanceF1 = instanceDenom > 0.0 ? 2.0 * r.finalOKCount / instanceDenom : 0.0;
}

void evaluateOneImageMerged(const string& imageId,
    const vector<EvalParticle>& gt,
    const vector<EvalParticle>& pred,
    EvalImageResult& result,
    vector<EvalMatchResult>& matches,
    vector<pair<int, vector<int>>>& under,
    vector<pair<int, vector<int>>>& over)
{
    result = EvalImageResult(); result.imageId = imageId;
    result.gtCount = int(gt.size()); result.predCount = int(pred.size());
    matches.clear(); under.clear(); over.clear();
    const int G = int(gt.size()), P = int(pred.size());
    vector<vector<double>> inter(G, vector<double>(P, 0.0));
    vector<EvalPairCandidate> candidates;
    for (int i = 0; i < G; ++i) for (int j = 0; j < P; ++j) {
        double intersection = evalIntersectionArea(gt[i].mask, pred[j].mask);
        inter[i][j] = intersection;
        double uni = gt[i].areaPx + pred[j].areaPx - intersection;
        double iou = (uni > 1e-6) ? intersection / uni : 0.0;
        if (iou >= MATCH_IOU_THRESH) candidates.push_back({ i,j,intersection,iou });
    }
    
    stable_sort(candidates.begin(), candidates.end(), [](const EvalPairCandidate& a, const EvalPairCandidate& b) {return a.iou > b.iou; });
    vector<int> gtMatched(G, 0), predMatched(P, 0);
    for (const auto& c : candidates) {
        if (gtMatched[c.gtIdx] || predMatched[c.predIdx]) continue;
        gtMatched[c.gtIdx] = predMatched[c.predIdx] = 1;
        EvalMatchResult m;
        m.gtIdx = c.gtIdx; m.predIdx = c.predIdx;
        m.interArea = c.interArea; m.gtArea = gt[c.gtIdx].areaPx; m.predArea = pred[c.predIdx].areaPx;
        m.iou = c.iou;
        if (m.gtArea + m.predArea > 1e-6) m.dice = 2.0 * m.interArea / (m.gtArea + m.predArea);
        if (m.predArea > 1e-6) m.precision = m.interArea / m.predArea;
        if (m.gtArea > 1e-6) m.recall = m.interArea / m.gtArea;
        m.iouOK = m.iou >= FINAL_IOU_THRESH;
        m.gtLongDiameter = gt[c.gtIdx].feretLongDiameter;
        m.predLongDiameter = pred[c.predIdx].feretLongDiameter;
        if (m.gtLongDiameter > 1e-6) m.longRelErr = fabs(m.predLongDiameter - m.gtLongDiameter) / m.gtLongDiameter;
        m.longDiameterOK = m.longRelErr <= FERET_LONG_REL_ERR_THRESH;
        m.finalOK = m.iouOK && m.longDiameterOK;
        matches.push_back(m);
    }
    result.matchedCount = int(matches.size());
    for (const auto& m : matches) {
        result.sumIou += m.iou; result.sumDice += m.dice;
        result.sumPrecision += m.precision; result.sumRecall += m.recall;
        result.sumLongRelErr += m.longRelErr;
        if (m.iouOK) ++result.iouOKCount; else ++result.iouBadCount;
        if (m.longDiameterOK) ++result.longOKCount; else ++result.longBadCount;
        if (m.finalOK) ++result.finalOKCount;
    }
    for (int j = 0; j < P; ++j) {
        vector<int> covered;
        for (int i = 0; i < G; ++i) {
            if (gt[i].areaPx > 0 && inter[i][j] / gt[i].areaPx > UNDER_GT_COVER_THRESH) covered.push_back(i);
        }
        if (covered.size() >= 2) { ++result.underSegPredCount; under.push_back({ j,covered }); }
    }
    for (int i = 0; i < G; ++i) {
        vector<int> inside;
        for (int j = 0; j < P; ++j) {
            if (pred[j].areaPx > 0 && inter[i][j] / pred[j].areaPx > OVER_PRED_INSIDE_THRESH) inside.push_back(j);
        }
        if (inside.size() >= 2) { ++result.overSegGtCount; over.push_back({ i,inside }); }
    }
    for (int i = 0; i < G; ++i) if (!gtMatched[i]) ++result.missedGtCount;
    for (int j = 0; j < P; ++j) if (!predMatched[j]) ++result.falsePredCount;
    evalFinalizeMetrics(result);
}

void evalAddToTotal(EvalImageResult& total, const EvalImageResult& r)
{
    total.gtCount += r.gtCount; total.predCount += r.predCount; total.matchedCount += r.matchedCount;
    total.iouOKCount += r.iouOKCount; total.iouBadCount += r.iouBadCount;
    total.longOKCount += r.longOKCount; total.longBadCount += r.longBadCount;
    total.finalOKCount += r.finalOKCount; total.underSegPredCount += r.underSegPredCount;
    total.overSegGtCount += r.overSegGtCount; total.missedGtCount += r.missedGtCount;
    total.falsePredCount += r.falsePredCount;
    total.sumIou += r.sumIou; total.sumDice += r.sumDice;
    total.sumPrecision += r.sumPrecision; total.sumRecall += r.sumRecall;
    total.sumLongRelErr += r.sumLongRelErr;
}
void evalFinalizeTotal(EvalImageResult& total)
{
    evalFinalizeMetrics(total);
}

void evalWriteUTF8BOM(ofstream& ofs)
{
    const unsigned char bom[] = { 0xEF,0xBB,0xBF };
    ofs.write(reinterpret_cast<const char*>(bom), 3);
}

struct EvalCsvWriter {
    ofstream summary;
    EvalImageResult total;
    int evaluatedImages = 0, skippedImages = 0;

    explicit EvalCsvWriter(const fs::path& outputDir)
    {
        summary.open(outputDir / "eval_summary.csv", ios::binary);
        if (!summary)
            throw runtime_error("Failed to create evaluation summary CSV: " + (outputDir / "eval_summary.csv").string());
        evalWriteUTF8BOM(summary);
        summary << fixed << setprecision(6);
        summary << "Image_ID,GT_Count,Pred_Count,Matched_Count,IoU_Pass_Count,IoU_Fail_Count,"
                << "Feret_Pass_Count,Feret_Fail_Count,Successful_Count,Undersegmented_Pred_Count,"
                << "Oversegmented_GT_Count,Missed_GT_Count,False_Pred_Count,Mean_IoU,"
                << "Mean_Matched_Dice,Mean_Matched_Pixel_Precision,Mean_Matched_Pixel_Recall,"
                << "Mean_Feret_Relative_Error_With_Penalty,Particle_Segmentation_Success_Rate,"
                << "Instance_Precision,Instance_Recall,Instance_F1,Evaluation_Item_Count,"
                << "Instance_FP,Instance_FN\n";
    }

    void logStatus(const string&, bool ok, const string&) {
        if (!ok) ++skippedImages;
    }

    void writeSummary(const string& id, const EvalImageResult& r) {
        summary << id << ',' << r.gtCount << ',' << r.predCount << ',' << r.matchedCount << ','
            << r.iouOKCount << ',' << r.iouBadCount << ',' << r.longOKCount << ','
            << r.longBadCount << ',' << r.finalOKCount << ',' << r.underSegPredCount << ','
            << r.overSegGtCount << ',' << r.missedGtCount << ',' << r.falsePredCount << ','
            << r.avgIou << ',' << r.avgDice << ',' << r.avgPrecision << ',' << r.avgRecall << ','
            << r.avgLongRelErr << ',' << r.accuracy << ',' << r.instancePrecision << ','
            << r.instanceRecall << ',' << r.instanceF1 << ',' << r.evaluationItemCount << ','
            << r.instanceFalsePositiveCount << ',' << r.instanceFalseNegativeCount << '\n';
    }

    void append(const string& id, const EvalImageResult& r) {
        ++evaluatedImages;
        evalAddToTotal(total, r);
        writeSummary(id, r);
    }

    void finish() {
        evalFinalizeTotal(total);
        writeSummary("Total", total);
        summary.flush();
    }
};

fs::path findProjectRoot(const fs::path& programPath)
{
    vector<fs::path> locations = { fs::current_path() };
#ifdef _WIN32
    wchar_t executablePath[32768] = {};
    const DWORD length = GetModuleFileNameW(nullptr, executablePath, 32768);
    if (length > 0 && length < 32768)
        locations.push_back(fs::path(executablePath).parent_path());
#endif
    if (!programPath.empty())
        locations.push_back(fs::absolute(programPath).parent_path());
    locations.push_back(fs::absolute(fs::path(__FILE__)).parent_path());

    for (const fs::path& location : locations)
    {
        fs::path current = fs::absolute(location);
        while (!current.empty())
        {
            if (fs::is_directory(current / "data"))
                return current;
            if (current == current.root_path())
                break;
            current = current.parent_path();
        }
    }
    return {};
}

int runTestSubset(const fs::path& inputImageDir,
    const fs::path& inputGtDir, const fs::path& outRoot)
{
    using namespace std::chrono;
    const vector<fs::path> imageFiles = getEvalImageFiles(inputImageDir);
    if (imageFiles.empty()) {
        cerr << "No matching input images found: " << inputImageDir.string() << '\n';
        return -1;
    }
    set<string> seenStems;
    for (const auto& file : imageFiles) {
        const string stem = file.stem().string();
        if (!seenStems.insert(stem).second) {
            cerr << "Duplicate image basename detected: " << stem
                << "; keep only one image per ID to avoid overwriting JSON and double-counting.\n";
            return -1;
        }
    }
    const fs::path outDir = outRoot;
    const vector<string> legacyCsvFiles = {
        "eval_detail.csv",
        "gt_feret_long_detail.csv",
        "true_particle_size_detail.csv",
        "pred_particle_size_detail.csv",
        "true_particle_size_distribution.csv",
        "pred_particle_size_distribution.csv",
        "evaluation_input_status.csv"
    };
    for (const string& filename : legacyCsvFiles) {
        error_code ec;
        fs::remove(outDir / filename, ec);
        if (ec) cerr << "Warning: could not remove obsolete CSV "
                     << (outDir / filename).string() << ": " << ec.message() << '\n';
    }
    const fs::path jsonOutDir = outDir / "predicted_json";
    const fs::path finalImagesDir = outDir / "final_segmentation";
    fs::create_directories(jsonOutDir);
    fs::create_directories(finalImagesDir);
    cout << "Input images: " << inputImageDir.string() << '\n';
    cout << "Ground-truth annotations: " << inputGtDir.string() << '\n';
    cout << "Expected image IDs: 1-200 (.jpg only); available images: " << imageFiles.size() << "\n";
    cout << "Output directory: " << outRoot.string() << '\n';

    const fs::path timeCsvPath = outDir / "algorithm_inference_time.csv";
    ofstream timeCsv(timeCsvPath, ios::binary);
    if (!timeCsv) { cerr << "Failed to create algorithm timing CSV: " << timeCsvPath.string() << '\n'; return -1; }
    evalWriteUTF8BOM(timeCsv);
    timeCsv << fixed << setprecision(6);
    timeCsv << "Image_ID,Algorithm_Time_ms\n";
    EvalCsvWriter evalCsv(outDir);
    double totalAlgorithmTimeMs = 0.0;
    int validImageCount = 0;
    const vector<int> pngParams = { IMWRITE_PNG_COMPRESSION, 1 };

    for (const fs::path& imgPath : imageFiles) {
        const string stem = imgPath.stem().string();
        Mat src = imread(imgPath.string(), IMREAD_COLOR);
        if (src.empty()) {
            cerr << "Failed to read image: " << imgPath.string() << '\n';
            evalCsv.logStatus(stem, false, "Image read failed");
            continue;
        }
        const auto algorithmStart = steady_clock::now();
        vector<vector<Point>> predContoursGlobal;
        vector<pair<Point, Point>> finalSegLinesGlobal;
        CenterPeakStats peakStats;
        Mat gray;
        cvtColor(src, gray, COLOR_BGR2GRAY);
        Mat binary, flipped;
        threshold(gray, binary, BIN_THRESH, 255, THRESH_BINARY);
        bitwise_not(binary, flipped);
        const int smallHoleFillArea = 200;
        Mat filledSmallHoles = fillSmallInternalHoles(flipped, smallHoleFillArea);
        Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
        Mat morph;
        morphologyEx(filledSmallHoles, morph, MORPH_CLOSE, kernel, Point(-1, -1), 0);
        Mat labels, stats, centroids;
        const int nLabels = connectedComponentsWithStats(morph, labels, stats, centroids, 8);

        const int roiPad = 5;
        int processedComponentCount = 0;
        for (int label = 1; label < nLabels; ++label) {
            const int area = stats.at<int>(label, CC_STAT_AREA);
            if (area < MIN_AREA) continue;
            const int left = max(0, stats.at<int>(label, CC_STAT_LEFT) - roiPad);
            const int top = max(0, stats.at<int>(label, CC_STAT_TOP) - roiPad);
            const int right = min(src.cols, stats.at<int>(label, CC_STAT_LEFT) + stats.at<int>(label, CC_STAT_WIDTH) + roiPad);
            const int bottom = min(src.rows, stats.at<int>(label, CC_STAT_TOP) + stats.at<int>(label, CC_STAT_HEIGHT) + roiPad);
            const Rect roi(left, top, right - left, bottom - top);
            if (roi.width <= 0 || roi.height <= 0) continue;
            Mat mask = (labels(roi) == label);
            mask.convertTo(mask, CV_8U, 255);
            processConnectedComponent(mask, roi,
                predContoursGlobal, finalSegLinesGlobal, peakStats);
            ++processedComponentCount;
        }
        const auto algorithmEnd = steady_clock::now();
        const double timeMs = duration<double, milli>(algorithmEnd - algorithmStart).count();
        totalAlgorithmTimeMs += timeMs;
        ++validImageCount;
        timeCsv << stem << ',' << timeMs << '\n';

        const Mat finalSeg = drawFinalSegLinesOnOriginal(src, finalSegLinesGlobal);
        imwrite((finalImagesDir / ("final_seg_on_original_fast_" + stem + ".png")).string(), finalSeg, pngParams);
        const fs::path predJsonPath = jsonOutDir / (stem + ".json");
        { error_code ec; fs::remove(predJsonPath, ec); }
        bool predSaved = savePredContoursToLabelmeJson(imgPath.string(), predJsonPath.string(),
            predContoursGlobal, src.cols, src.rows);
        if (!predSaved) {
            evalCsv.logStatus(stem, false, "Failed to save predicted JSON; older results will not be reused");
            continue;
        }
        const fs::path gtJsonPath = inputGtDir / (stem + ".json");
        vector<EvalParticle> gtParticles, predParticles;
        string err;
        if (!loadEvalParticles(gtJsonPath, src.size(), gtParticles, err)) {
            evalCsv.logStatus(stem, false, "GT: " + err);
            cout << "Image " << stem << " could not be evaluated: " << err << '\n';
            continue;
        }
        if (!loadEvalParticles(predJsonPath, src.size(), predParticles, err)) {
            evalCsv.logStatus(stem, false, "Pred: " + err);
            cout << "Image " << stem << " could not be evaluated: " << err << '\n';
            continue;
        }
        EvalImageResult res;
        vector<EvalMatchResult> matches;
        vector<pair<int, vector<int>>> under, over;
        evaluateOneImageMerged(stem, gtParticles, predParticles, res, matches, under, over);
        evalCsv.append(stem, res);
        cout << fixed << setprecision(4)
            << "Image " << stem << " | GT=" << res.gtCount << " Pred=" << res.predCount
            << " Matched=" << res.matchedCount << " Successful=" << res.finalOKCount
            << " SuccessRate=" << res.accuracy * 100.0 << "% MeanIoU=" << res.avgIou
            << " PenalizedFeretRelativeError=" << res.avgLongRelErr * 100.0
            << "% InstanceP/R/F1=" << res.instancePrecision << '/' << res.instanceRecall << '/' << res.instanceF1
            << " AlgorithmTime=" << timeMs << "ms\n";
    }
    const double avgMs = validImageCount > 0 ? totalAlgorithmTimeMs / validImageCount : 0.0;
    timeCsv.close();
    evalCsv.finish();
    const EvalImageResult& tot = evalCsv.total;
    cout << fixed << setprecision(4)
        << "Images=" << imageFiles.size()
        << " Processed=" << validImageCount
        << " Evaluated=" << evalCsv.evaluatedImages
        << " SkippedEvaluation=" << evalCsv.skippedImages << '\n'
        << "Total GT=" << tot.gtCount << " Pred=" << tot.predCount
        << " Matched=" << tot.matchedCount << " Successful=" << tot.finalOKCount
        << " Undersegmented=" << tot.underSegPredCount << " Oversegmented=" << tot.overSegGtCount
        << " Missed=" << tot.missedGtCount << " FalsePred=" << tot.falsePredCount << '\n'
        << "OverallSuccessRate=" << tot.accuracy * 100.0 << "% MeanIoU=" << tot.avgIou
        << " MeanMatchedDice=" << tot.avgDice
        << " MeanPenalizedFeretRelativeError=" << tot.avgLongRelErr * 100.0 << "%" << '\n'
        << "InstancePrecision=" << tot.instancePrecision << " Recall=" << tot.instanceRecall << " F1=" << tot.instanceF1 << '\n'
        << "MeanAlgorithmTime=" << avgMs << "ms | Summary: " << (outDir / "eval_summary.csv").string() << '\n';
    cout << "\nImage processing, JSON output, and evaluation finished. Output directory: " << outRoot.string() << '\n';
    return 0;
}

int main(int argc, char* argv[])
{
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
#endif
    const fs::path projectRoot = findProjectRoot(argc > 0 ? fs::path(argv[0]) : fs::path());
    if (projectRoot.empty()) {
        cerr << "Project root not found. Create a data directory inside the project root.\n";
        return -1;
    }

    const fs::path testRoot = projectRoot / "data";
    const fs::path resultsRoot = projectRoot / "results";
    const vector<string> subsets = { "two_particle", "three_particle", "ring_particle" };
    int processedSubsets = 0;
    for (const string& subset : subsets)
    {
        const fs::path inputImageDir = testRoot / subset / "images";
        const fs::path inputGtDir = testRoot / subset / "annotations";
        if (!fs::is_directory(inputImageDir) && !fs::is_directory(inputGtDir))
            continue;
        if (!fs::is_directory(inputImageDir) || !fs::is_directory(inputGtDir)) {
            cerr << "Incomplete dataset folder: " << (testRoot / subset).string() << '\n';
            return -1;
        }
        bool hasJpg = false;
        for (const auto& entry : fs::directory_iterator(inputImageDir)) {
            if (entry.is_regular_file() && entry.path().extension() == ".jpg") {
                hasJpg = true;
                break;
            }
        }
        if (!hasJpg)
            continue;

        cout << "\n========== " << subset << " ==========\n";
        if (runTestSubset(inputImageDir, inputGtDir, resultsRoot / subset) != 0)
            return -1;
        ++processedSubsets;
    }
    if (processedSubsets == 0) {
        cerr << "No test dataset found. Create two_particle, three_particle, and ring_particle under data, "
             << "each containing images and annotations.\n";
        return -1;
    }
    cout << "\nFinished processing " << processedSubsets << " test dataset categories.\n";
    return 0;
}
