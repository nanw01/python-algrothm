/* Write your T-SQL query statement below */
with dates as (
	select s_date = min(period_start),
		e_date = max(period_end)
	from sales
	union all
	select dateadd(day, 1, s_date),
		e_date
	from dates
	where s_date < e_date
)
select PRODUCT_ID = cast(p.product_id as varchar(200)),
	PRODUCT_NAME = p.product_name,
	REPORT_YEAR = cast(year(s_date) as varchar(10)),
	TOTAL_AMOUNT = sum(average_daily_sales)
from product p
	left outer join sales s on p.product_id = s.product_id
	left outer join dates d on d.s_date between s.period_start and s.period_end
group by p.product_id,
	p.product_name,
	year(s_date)
order by 1,
	3 option(maxrecursion 0);
--
--
select t2.product_id,
	t3.product_name,
	t1.report_year,
	(
		datediff(
			if(
				t1.end_date < t2.period_end,
				t1.end_date,
				t2.period_end
			),
			# 结束时间取最小
			if(
				t1.start_date > t2.period_start,
				t1.start_date,
				t2.period_start
			) # 开始时间取最大
		) + 1
	) * t2.average_daily_sales as total_amount
from (
		select '2018' as report_year,
			date('2018-01-01') as start_date,
			date('2018-12-31') as end_date
		from dual
		union all
		select '2019' as year,
			date('2019-01-01') as start_date,
			date('2019-12-31') as end_date
		from dual
		union all
		select '2020' as year,
			date('2020-01-01') as start_date,
			date('2020-12-31') as end_date
		from dual
	) t1,
	Sales t2,
	Product t3
where (
		# 销售区间包含自然年的后半段
		(
			t2.period_start between t1.start_date and t1.end_date
		) # 销售区间包含自然年的前半段
		or (
			t2.period_end between t1.start_date and t1.end_date
		) # 整个自然年都在销售区间之内
		or (
			t1.start_date > t2.period_start
			and t1.end_date < t2.period_end
		)
	)
	and t2.product_id = t3.product_id
order by product_id,
	report_year;
--
--
select s.product_id,
	p.product_name,
	d.report_year,
	(
		datediff(
			if(
				d.period_end < s.period_end,
				d.period_end,
				s.period_end
			),
			if(
				d.period_start > s.period_start,
				d.period_start,
				s.period_start
			)
		) + 1
	) * s.average_daily_sales as total_amount
from Product as p,
	Sales as s,
	(
		select '2018' as report_year,
			date('2018-01-01') as period_start,
			date('2018-12-31') as period_end
		from dual
		union all
		select '2019' as report_year,
			date('2019-01-01') as period_start,
			date('2019-12-31') as period_end
		from dual
		union all
		select '2020' as report_year,
			date('2020-01-01') as period_start,
			date('2020-12-31') as period_end
		from dual
	) as d
where p.product_id = s.product_id
	and (
		(
			s.period_start between d.period_start and d.period_end
		)
		or (
			s.period_end between d.period_start and d.period_end
		)
		or(
			d.period_start > s.period_start
			and d.period_end < s.period_end
		)
	)
order by product_id,
	report_year