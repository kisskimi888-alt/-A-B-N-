# E-mail Campaign Test: What to Send, and to Whom

**To:** Product / CRM lead · **Data:** 64,000 customers, randomized into 3 equal groups, 2-week outcome window

## Recommendation

**Send the Mens e-mail as the default campaign. Stop sending the Womens e-mail to customers who have only
bought mens products. Use a simple segment rule for targeting, not an ML model.**

## What we tested

Customers were randomly assigned to get the Mens e-mail, the Womens e-mail, or no e-mail. The groups
were balanced: sizes matched the 1:1:1 design (p = 0.90) and pre-test customer profiles were nearly
identical. So differences in outcomes are caused by the e-mails.

## Results (vs. no e-mail)

| | Visit rate | Conversion rate | Revenue per customer |
|---|---|---|---|
| No e-mail (baseline) | 10.6% | 0.57% | $0.65 |
| **Mens e-mail** | **+7.7 pts (+72%)** | **+0.68 pts (+119%)** | **+$0.77 (+118%)** |
| Womens e-mail | +4.5 pts (+43%) | +0.31 pts (+54%) | +$0.42 (+65%) |

- Every effect is statistically significant, even after correcting for testing 6 comparisons.
- The Mens e-mail beats the Womens e-mail on visits (+3.1 pts), conversions (+0.37 pts) and revenue (+$0.35).
  The revenue gap is the least certain (95% range: +$0.03 to +$0.66).
- The revenue lift comes from **more people buying**, not bigger orders. Average order size is about
  $114–122 in every group.

**At scale** (if these effects hold): per 100,000 customers, the Mens e-mail adds about **7,700 visits,
680 orders and $77,000 in revenue** (95% range: $49,000–$105,000) over two weeks.

## Who responds

| Customer's past purchases | Mens e-mail | Womens e-mail |
|---|---|---|
| Mens products only | +6.9 pts visits | **+1.1 pts** (nearly no effect) |
| Womens products only | +7.1 pts | +7.4 pts (tie with Mens) |
| Both | +13.4 pts | +7.1 pts |

The Womens e-mail barely moves customers who have only bought mens products. The Mens e-mail works
across all three groups.

We also trained five machine-learning uplift models to pick who should get the Womens e-mail. The
best one did **no better than the rule "send it only to past womens buyers."** Both reach 66% of the
e-mail's total effect by e-mailing 40% of customers. The rule is free to run, easy to explain, and
cannot silently break. For the Mens e-mail, no model found a group worth excluding.

## What we are less sure about

- **Profitability.** We see revenue, not margin or send cost. Sending pays off when
  *cost per e-mail < margin × $0.77*. For example, at an assumed 30% margin, an e-mail must cost less than $0.23.
- **Long-term effects.** We measured 2 weeks only. Unsubscribes and e-mail fatigue were not tracked.
- **Revenue precision.** Only ~580 customers bought anything, so revenue estimates have wide ranges.
- **Age of data.** The test ran in 2008. Re-validate before relying on the exact numbers.

## Suggested next step

Run a follow-up test of **Mens e-mail to everyone** vs. **Mens e-mail, with the Womens e-mail for
womens-only buyers**. Track unsubscribes, and run long enough to measure revenue: the current test can
only reliably detect revenue changes of about 48% or more.

*Full methods, code and robustness checks: see the project README.*
